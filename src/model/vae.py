
# Referenced/modified code from Shijie Wu's crosslingual-nlp repository,
#   particularly `model/tagger.py`.
# Taken code includes basic class structure, imports, and method headers,
# such as initialization code.
# Changed forward to __call__.

import torch.nn.functional as F
import yaml
from argparse import Namespace
import os
import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import math

from metric import LABEL_PAD_ID
from model import BaseVAE, Model, Tagger
from enumeration import Split
import constant
import util

import wandb

class VAE(BaseVAE):
    
    def __init__(self, hparams):
        super(VAE, self).__init__(hparams)
        prior_types = {
            'optimized_data',
            'fixed_data',
            'fixed_uniform',
        }
        
        # mBERT initialization, random decoder initialization is handled by BaseVAE initialization.
        if self.hparams.decoder_checkpoint:
            base_decoder = BaseVAE.load_from_checkpoint(self.hparams.decoder_checkpoint)
            # Overwrite all of the attributes
            self.decoder = base_decoder.decoder
            if base_decoder.use_auxiliary:
                if not self.hparams.auxiliary_size > 0: print('Overriding hparams on VAE level, always following BaseVAE architecture.')
                self.use_auxiliary = base_decoder.use_auxiliary
                self.auxiliary_mu = base_decoder.auxiliary_mu
            # Set by mBERT type (if that not present: argument), which is set by base internally.
            assert base_decoder.concat_all_hidden_states == self.concat_all_hidden_states,\
                f'Base decoder: {base_decoder.concat_all_hidden_states}, self: {self.concat_all_hidden_states}'
            
        # Encoder initialization
        if self.hparams.encoder_checkpoint:
            base_encoder = Tagger.load_from_checkpoint(self.hparams.encoder_checkpoint)
            self.classifier = base_encoder.classifier
            assert base_encoder.concat_all_hidden_states == self.concat_all_hidden_states,\
                f'Base encoder: {base_encoder.concat_all_hidden_states}, self: {self.concat_all_hidden_states}'
        else:
            self.classifier = self.build_classifier()
    
        smoothed_english_prior = self.get_smoothed_english_prior()
        if self.hparams.prior_type == 'optimized_data':
            self.prior_param = torch.nn.ParameterDict()
            self.prior_param['log_prior'] = torch.nn.Parameter(torch.log(smoothed_english_prior), requires_grad = True)
            self.loss_prior = self.prior_param.log_prior
        elif self.hparams.prior_type == 'fixed_data':
            self.loss_prior = util.apply_gpu(smoothed_english_prior)
        elif self.hparams.prior_type == 'fixed_uniform':
            self.loss_prior = util.apply_gpu(self.get_uniform_prior())
        else:
            assert self.hparams.prior_type in prior_types, f"Choose prior type from: {prior_types}. Current parameter: {self.hparams.prior_type}"

        checkpoint_metric = 'acc' if self.hparams.encoder_checkpoint else 'nmi_added'
        self._selection_criterion = f'val_{self.target_language}_{checkpoint_metric}_epoch_monitor'
        self._comparison_mode = 'max'
        self.optimization_loss = 'vae_loss'
        self.metric_names = [
            'loss_KL',
            'decoder_loss',
            'vae_loss',
            'encoder_loss',
            'MSE',
            'acc',
            'nmi'
        ]
        if self.use_auxiliary:
            self.metric_names.append('auxiliary_KL')
        self.setup_metrics()
        
    def get_uniform_prior(self):
        number_of_labels = len(constant.UD_POS_LABELS)
        return torch.ones(number_of_labels) / number_of_labels
        
    def get_prior(self, lang, split):
        counts = self.get_label_counts(lang, split)
        return counts / torch.sum(counts)
    
    def get_smoothed_prior(self, lang, split):
        threshold = 0.001
        prior = self.get_prior(lang, split)
        raw_smoothed_prior = torch.clamp(prior, min = threshold)
        assert torch.all(raw_smoothed_prior >= threshold), raw_smoothed_prior
        return raw_smoothed_prior / torch.sum(raw_smoothed_prior)
        
    def get_smoothed_english_prior(self):
        prior = self.get_smoothed_prior('English', Split.train)
        return prior
        
    def calculate_log_pi_t(self, batch, hs):
        logits = self.classifier(hs)
        raw_log_pi_t = F.log_softmax(logits, dim=-1)
        # Need to remove the log_pi_t that do not correspond to real inputs
        log_pi_t = self.set_padded_to_zero(batch, raw_log_pi_t)
        return log_pi_t
        
    def calculate_kl_against_prior(self, batch, log_q_given_input, prior):
        repeated_prior = prior.reshape(1, 1, prior.shape[0]).repeat(log_q_given_input.shape[0], log_q_given_input.shape[1], 1)
        raw_pre_sum = torch.exp(log_q_given_input) * (log_q_given_input - torch.log(repeated_prior))
        kl_divergence_mean = self.calculate_clean_metric(batch, raw_pre_sum)
        return kl_divergence_mean

    def calculate_encoder_loss(self, batch, log_pi_t):
        encoder_loss = F.nll_loss(
            log_pi_t.view(-1, self.nb_labels),
            batch["labels"].view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        return encoder_loss
        
    def calculate_one_hot_from_probabilities(self, pi_tilde_t):
        predicted_labels = pi_tilde_t.argmax(dim=-1)
        one_hot_predictions = F.one_hot(predicted_labels, num_classes = self.nb_labels)
        assert pi_tilde_t.shape == one_hot_predictions.shape, f'pi_tilde_t: {pi_tilde_t.shape}, one_hot: {one_hot_predictions.shape}'
        return one_hot_predictions
    
    def __call__(self, batch):
        self.model.eval()
        current_language = batch['lang'][0]
        assert not any(list(filter(lambda example : example != current_language, batch['lang'])))
       
        hs = self.calculate_hidden_states(batch)
        log_pi_t = self.calculate_log_pi_t(batch, hs)
        
        # Calculate predicted mean
        assert self.classifier.training == self.decoder.training
        is_train = self.classifier.training
        if self.hparams.pos_kl_weight > 0 and is_train:
            uniform_sample = Uniform(torch.zeros(log_pi_t.shape), torch.ones(log_pi_t.shape)).rsample()
            noise = util.apply_gpu(-torch.log(-torch.log(uniform_sample)))
        
            unnormalized_pi_tilde_t = (log_pi_t + noise) / self.hparams.temperature
            raw_pi_tilde_t = F.softmax(unnormalized_pi_tilde_t, dim=-1)
        else:
            raw_pi_tilde_t = torch.exp(log_pi_t)
        
        # For eval, do NOT add noise, but DO convert to one-hot representation
        if not is_train:
            # Need float for compatibility in cleaning later (as model input)
            raw_pi_tilde_t = self.calculate_one_hot_from_probabilities(raw_pi_tilde_t).float()
            
        pi_tilde_t = self.set_padded_to_zero(batch, raw_pi_tilde_t)
        loss = self.calculate_decoder_loss(batch, hs, pi_tilde_t)
    
        # Only softmax if the prior is optimized (otherwise it's already probability)
        prior = self.loss_prior if self.hparams.prior_type != 'optimized_data' else self.loss_prior.softmax(dim=-1)
        loss['loss_KL'] = self.calculate_kl_against_prior(batch, log_pi_t, prior)
        loss['decoder_loss'] = self.hparams.mse_weight * loss['MSE'] + self.hparams.pos_kl_weight * loss['loss_KL']
        if self.use_auxiliary:
            loss['decoder_loss'] += self.hparams.auxiliary_kl_weight * loss['auxiliary_KL']
        loss['vae_loss'] = loss['decoder_loss']
        with torch.no_grad():
            loss['encoder_loss'] = self.calculate_encoder_loss(batch, log_pi_t)
        
        if math.isnan(loss['vae_loss']): import pdb; pdb.set_trace()
        self.add_language_to_batch_output(loss, batch)
        return loss, log_pi_t
        
    def step_helper(self, batch, prefix):
        return Model.step_helper(self, batch, prefix)
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = BaseVAE.add_model_specific_args(parser)
        parser.add_argument("--encoder_checkpoint", default='', type=str)
        parser.add_argument("--decoder_checkpoint", default='', type=str)
        parser.add_argument("--pos_kl_weight", default=1, type=float)
        
        parser.add_argument("--pos_nll_weight", default=0, type=float)
        parser.add_argument("--temperature", default=1, type=float)
        parser.add_argument("--prior_type", default='optimized_data', type=str)
        return parser
    
    def train_dataloader(self):
        return super().train_dataloader()
        
        