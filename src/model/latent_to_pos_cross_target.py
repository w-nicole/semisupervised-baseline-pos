# Referenced/modified code from Shijie Wu's crosslingual-nlp repository,
#   particularly `model/tagger.py`.
# Taken code includes basic class structure, imports, and method headers,
# general method logic for all methods that also exist in `tagger.py`,
#   excluding new functionality per new decoder logic/irrelevant old functionality
#   as well as inheritance-related code (such as the declaration of the argparse)
#   or code that directly overrides previous code
#  (like the mix_sampling default value,
#  the logic of the forward/training_step code was maintained with irrelevant code removed, with new logic added)  
#  Most other code was added.
# See LICENSE in this codebase for license information.

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from metric import LABEL_PAD_ID
from dataset import tagging, collate
from model.base import Model
from model.module import LSTMLinear
from model.tagger import Tagger
from model.base_tagger import BaseTagger
from enumeration import Split
import util
import constant

class LatentToPOSCross(BaseTagger):
    
    def __init__(self, hparams):
        super(LatentToPOSCross, self).__init__(hparams)
        load_tagger = lambda checkpoint : Tagger.load_from_checkpoint(checkpoint) if checkpoint else Tagger(self.hparams)
        # Target
        target_tagger = load_tagger(self.hparams.target_mbert_checkpoint)
        self.freeze_bert(target_tagger)
        self.target_mbert = target_tagger.model
        # Encoder
        encoder_tagger = load_tagger(self.hparams.encoder_mbert_checkpoint)
        # Below does not necessarily always have to be true -- if needs to be untrue, check the inner frozen/outer unfrozen case
        assert not (encoder_tagger.is_frozen_mbert and not self.hparams.freeze_mbert),\
            f"Inner: {encoder_tagger.freeze_mbert}, Outer: {self.hparams.freeze_mbert}"
        if self.hparams.freeze_mbert:
            self.freeze_bert(encoder_tagger)
        self.encoder_mbert = encoder_tagger.model
        # Check concat status
        self.concat_all_hidden_states = target_tagger.concat_all_hidden_states
        assert self.concat_all_hidden_states == encoder_tagger.concat_all_hidden_states,\
            f'target: {target_tagger.concat_all_hidden_states}, encoder: {encoder_tagger.concat_all_hidden_states}'

        self.encoder_mu = self.build_layer_stack(
            self.mbert_output_size, self.hparams.latent_size,
            self.hparams.encoder_mu_hidden_size, self.hparams.encoder_mu_hidden_layers
        )
        self.encoder_log_var = self.build_layer_stack(
            self.mbert_output_size, self.hparams.latent_size,
            self.hparams.encoder_log_var_hidden_size, self.hparams.encoder_log_var_hidden_layers
        )
        pos_model_args = (
            self.hparams.latent_size, self.nb_labels,
            self.hparams.pos_hidden_size, self.hparams.pos_hidden_layers
        )
        reconstruction_model_args = (
            self.hparams.latent_size, self.mbert_output_size,
            self.hparams.reconstruction_hidden_size, self.hparams.reconstruction_hidden_layers
        )
        
        self.model_type = {
            'lstm' : LSTMLinear,
            'fcn' : self.build_layer_stack,
        }
        self.decoder_pos = self.model_type[self.hparams.pos_model_type](*pos_model_args)
        self.decoder_reconstruction = self.model_type[self.hparams.reconstruction_model_type](*reconstruction_model_args)
        self.optimization_loss = 'total_loss'
        self._selection_criterion = f'val_{self.target_language}_acc_epoch'
        self._comparison_mode = 'max'
        self.metric_names = [
            'latent_KL',
            'pos_nll',
            'total_loss',
            'MSE',
            'acc'
        ]
        self.setup_metrics()
        
    # Below forward-related methods:
    # Shijie Wu's code, but with decoder logic added and irrelevant options removed,
    # and variables renamed for notation consistency.
    
    def calculate_hidden_states(self, mbert, batch):
        # Updated call arguments
        hs = self.encode_sent(mbert, batch["sent"], batch["start_indices"], batch["end_indices"], batch["lang"])
        return hs
        
    def get_non_pad_label_mask(self, labels, tensor):
        if len(tensor.shape) == 3:
            repeated_labels = labels.unsqueeze(2).repeat(1, 1, tensor.shape[-1])
        else:
            repeated_labels = labels.unsqueeze(1).repeat(1, tensor.shape[-1])
        return util.apply_gpu(repeated_labels != LABEL_PAD_ID)        
        
    def set_padded_to_zero(self, labels, tensor):
        clean_tensor = torch.where(
            self.get_non_pad_label_mask(labels, tensor),
            tensor, util.apply_gpu(torch.zeros(tensor.shape))
        )
        return clean_tensor
    
    def calculate_clean_metric(self, batch, raw_metric_tensor):
        non_pad_mask = self.get_non_pad_label_mask(batch['labels'], raw_metric_tensor)
        clean_metric_tensor = self.set_padded_to_zero(batch['labels'], raw_metric_tensor)
        
        # for KL divergence, but MSE doesn't trigger false positive
        metric_per_position = torch.sum(clean_metric_tensor, axis = -1)
        if not torch.all(metric_per_position >= 0):
            import pdb; pdb.set_trace()
            
        # Adjust scale to NOT divide out the hidden size representation.
        clean_average = torch.sum(clean_metric_tensor) / torch.sum(non_pad_mask) * raw_metric_tensor.shape[-1]
        return clean_average
        
    def calculate_masked_mse_loss(self, batch, padded_mu_t, padded_hs):
        raw_metric_tensor = torch.pow(padded_mu_t - padded_hs, 2)
        clean_mse = self.calculate_clean_metric(batch, raw_metric_tensor)
        return clean_mse
    
    def get_latent_distribution(self, latent_mean, latent_sigma):
        return Normal(latent_mean, latent_sigma)
        
    def calculate_latent_kl(self, batch, latent_mean, latent_sigma):
        latent_distribution = self.get_latent_distribution(latent_mean, latent_sigma)
        normal_prior = Normal(
            util.apply_gpu(torch.zeros(latent_mean.shape)),
            util.apply_gpu(torch.ones(latent_mean.shape))
        )
        raw_kl_pre_sum = torch.distributions.kl.kl_divergence(latent_distribution, normal_prior)
        return self.calculate_clean_metric(batch, raw_kl_pre_sum)
        
    def calculate_encoder_loss(self, batch, log_pi_t):
        encoder_loss = F.nll_loss(
            log_pi_t.view(-1, self.nb_labels),
            batch["labels"].view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        return encoder_loss
        
    def calculate_encoder_outputs(self, batch, latent_sample):
        decoder_output = self.decoder_pos(*self.get_decoder_args(self.decoder_pos, batch, latent_sample))
        pos_log_probs = F.log_softmax(decoder_output, dim = -1)
        loss = self.calculate_encoder_loss(batch, pos_log_probs)
        return pos_log_probs, loss
        
    def calculate_intermediates(self, batch):
        if self.hparams.freeze_mbert:
            self.encoder_mbert.eval()
        encoder_hs = self.calculate_hidden_states(self.encoder_mbert, batch)
        latent_mean = self.encoder_mu(encoder_hs)
        latent_sigma = torch.exp(self.encoder_log_var(encoder_hs)).sqrt()
        latent_sample = self.get_latent_distribution(latent_mean, latent_sigma).rsample()\
            if not self.hparams.debug_without_sampling else latent_mean
        return encoder_hs, latent_sample, latent_mean, latent_sigma
      
    def get_decoder_args(self, decoder, batch, latent_sample):
        return (batch, latent_sample) if isinstance(decoder, LSTMLinear) else (latent_sample,)
     
    # Changed from forward.
    def __call__(self, batch):
        current_language = batch['lang'][0]
        assert not any(list(filter(lambda example : example != current_language, batch['lang'])))
        
        loss = {} 
        encoder_hs, latent_sample, latent_mean, latent_sigma = self.calculate_intermediates(batch)
        
        predicted_hs = self.decoder_reconstruction(*self.get_decoder_args(self.decoder_reconstruction, batch, latent_sample))

        loss['latent_KL'] = self.calculate_latent_kl(batch, latent_mean, latent_sigma)
        with torch.no_grad():
            self.target_mbert.eval()
            target_hs = self.calculate_hidden_states(self.target_mbert, batch)
        loss['MSE'] = self.calculate_masked_mse_loss(batch, predicted_hs, target_hs)
        unlabeled_loss = self.hparams.latent_kl_weight * loss['latent_KL'] + self.hparams.mse_weight * loss['MSE']
        
        # Labeled case,
        # but if training on English alone, then English should be treated as unsupervised.
        labeled_case = current_language == constant.SUPERVISED_LANGUAGE and (len(self.hparams.trn_langs) > 1 or self.hparams.english_alone_as_supervised)
        if labeled_case:
            pos_log_probs, encoder_loss = self.calculate_encoder_outputs(batch, latent_sample)
            loss['total_loss'] = self.hparams.pos_nll_weight * encoder_loss + unlabeled_loss
        else:
            with torch.no_grad():
                pos_log_probs, encoder_loss = self.calculate_encoder_outputs(batch, latent_sample)
            loss['total_loss'] = unlabeled_loss
            
        loss['pos_nll'] = encoder_loss
        self.add_language_to_batch_output(loss, batch)
        return loss, pos_log_probs
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = Tagger.add_model_specific_args(parser)
        parser.add_argument('--encoder_mbert_checkpoint', default='', type=str)
        parser.add_argument('--target_mbert_checkpoint', default='', type=str)
        parser.add_argument("--latent_size", default=64, type=int)
        parser = Model.add_layer_stack_args(parser, 'encoder_mu')
        parser = Model.add_layer_stack_args(parser, 'encoder_log_var')
        parser = Model.add_layer_stack_args(parser, 'pos')
        parser = Model.add_layer_stack_args(parser, 'reconstruction')
        parser.add_argument('--reconstruction_model_type', default='lstm', type=str)
        parser.add_argument('--pos_model_type', default='lstm', type=str)
        parser.add_argument("--pos_nll_weight", default=1, type=float)
        parser.add_argument("--latent_kl_weight", default=1, type=float)
        parser.add_argument("--mse_weight", default=1, type=float)
        parser.add_argument("--english_alone_as_supervised", default=True, type=util.str2bool)
        parser.add_argument("--debug_without_sampling", default=False, type=util.str2bool)
        return parser