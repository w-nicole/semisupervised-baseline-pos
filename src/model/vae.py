
# Referenced/modified code from Shijie Wu's crosslingual-nlp repository,
#   particularly `model/tagger.py`.
# Taken code includes basic class structure, imports, and method headers,
# such as initialization code.

import torch.nn.functional as F
import yaml
from argparse import Namespace
import os
import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from metric import LABEL_PAD_ID

from model import BaseVAE
from enumeration import Split
import constant
import util

class VAE(BaseVAE):
    
    def __init__(self, hparams):
        super(VAE, self).__init__(hparams)
        decoder = BaseVAE.load_from_checkpoint(self.hparams.decoder_checkpoint)
        
        prior = self.get_smoothed_english_prior()
        self.prior_param = torch.nn.ParameterDict()
        self.prior_param['prior'] = torch.nn.Parameter(prior, requires_grad = True)
        
        assert len(self.hparams.val_langs) == 1, "Validation prior code currently only designed for one validation language."
        self.validation_prior = util.apply_gpu(self.get_smoothed_prior(self.hparams.val_langs[0], Split.dev))
        self._selection_criterion = 'val_acc'
        self._comparison_mode = 'max'
    
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
        
    def calculate_KL_against_prior(self, log_q_given_input, prior):
        
        repeated_prior = prior.unsqueeze(0).repeat(log_q_given_input.shape[0], 1)
        pre_sum = torch.exp(log_q_given_input) * (log_q_given_input - torch.log(repeated_prior))
        assert pre_sum.shape == log_q_given_input.shape, f'pre_sum: {pre_sum.shape}, q_given_input: {log_q_given_input.shape}'
        
        kl_divergence = torch.sum(pre_sum, axis = -1)
        return kl_divergence.mean()
        
        
    def forward(self, batch):
        
        current_language = batch['lang'][0]
        assert not any(list(filter(lambda example : example != current_language, batch['lang'])))
        
        # Labeled case
        if current_language == constant.SUPERVISED_LANGUAGE:
            return BaseVAE.forward(self, batch)
            
        # Unlabeled case
        hs = self.calculate_hidden_states(batch)
        log_pi_t = self.calculate_log_pi_t(batch, hs)
        # Calculate predicted mean
        uniform_sample = Uniform(torch.zeros(log_pi_t.shape), torch.ones(log_pi_t.shape)).rsample()
        noise = util.apply_gpu(-torch.log(-torch.log(uniform_sample)))
        
        unnormalized_pi_tilde_t = (log_pi_t + noise) / self.hparams.temperature
        pi_tilde_t = F.softmax(unnormalized_pi_tilde_t, dim=-1)
        mu_t = self.calculate_decoder(pi_tilde_t)

        loss = self.calculate_agnostic_loss(batch, mu_t, hs)
        with torch.no_grad():
            loss['encoder_loss'] = F.nll_loss(
                    log_pi_t.view(-1, self.nb_labels),
                    batch["labels"].view(-1),
                    ignore_index=LABEL_PAD_ID,
            )
            
        log_q_given_input = log_pi_t.sum(dim=1) 
        
        loss['KL'] = self.calculate_KL_against_prior(log_q_given_input, self.prior_param.prior)
        with torch.no_grad():
            loss['target_KL'] = self.calculate_KL_against_prior(log_q_given_input, self.validation_prior)
            
        loss['decoder_loss'] = loss['MSE'] + self.hparams.kl_weight * loss['KL']
        return loss, log_pi_t
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--kl_weight", default=1, type=float)
        return parser
    
    def train_dataloader(self):
        assert not self.hparams.mix_sampling, "This must be set to false for the batches to work."
        assert any([current_dataset.lang == constant.SUPERVISED_LANGUAGE for current_dataset in self.trn_datasets])
        return super().train_dataloader()
        
    def evaluation_step_helper(self, batch, prefix):
        
        loss, encoder_log_probs = self.forward(batch)
        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language"
        lang = batch["lang"][0]
        self.metrics[lang].add(batch["labels"], encoder_log_probs)
        return loss
        
        