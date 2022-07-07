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
from dataset import tagging
from model.base import Model
from model.tagger import Tagger
from model.base_tagger import BaseTagger
from enumeration import Split
import util
import constant

class LatentToPOS(BaseTagger):
    
    def __init__(self, hparams):
        super(LatentToPOS, self).__init__(hparams)
        if self.hparams.mbert_checkpoint:
            mbert = Tagger.load_from_checkpoint(self.hparams.mbert_checkpoint)
        else:
            mbert = Tagger(self.hparams)
        self.freeze_bert(mbert)
        # Overwrite base and tagger attributes so that encode_sent will function correctly
        self.model = mbert.model
        self.concat_all_hidden_states = mbert.concat_all_hidden_states
        
        decoder_input_size = self.nb_labels
        self.encoder = self.build_layer_stack(
            self.mbert_output_size, self.hparams.latent_size,
            self.hparams.encoder_hidden_size, self.hparams.encoder_hidden_layers,
            self.hparams.encoder_nonlinear_first
        )
        self.decoder_pos = self.build_layer_stack(
            self.hparams.latent_size, self.nb_labels,
            self.hparams.decoder_pos_hidden_size, self.hparams.decoder_pos_hidden_layers,
            self.hparams.decoder_pos_nonlinear_first
        )
        self.decoder_reconstruction = self.build_layer_stack(
            self.hparams.latent_size, self.nb_labels,
            self.hparams.decoder_reconstruction_hidden_size, self.hparams.decoder_reconstruction_hidden_layers,
            self.hparams.decoder_reconstruction_nonlinear_first
        )
        self.optimization_loss = 'total_loss'
        self._selection_criterion = f'val_{self.target_language}_acc_epoch_monitor'
        self._comparison_mode = 'max'
        self.metric_names = [
            'latent_KL',
            'pos_nll'
            'total_loss',
            'MSE',
            'acc'
        ]
        self.setup_metrics()
        
    # Below forward-related methods:
    # Shijie Wu's code, but with decoder logic added and irrelevant options removed,
    # and variables renamed for notation consistency.
    
    def calculate_hidden_states(self, batch):
        # Updated call arguments
        hs = self.encode_sent(batch["sent"], batch["start_indices"], batch["end_indices"], batch["lang"])
        return hs
        
    def get_non_pad_label_mask(self, batch, tensor):
        repeated_labels = batch['labels'].unsqueeze(2).repeat(1, 1, tensor.shape[-1])
        return repeated_labels != LABEL_PAD_ID        
        
    def set_padded_to_zero(self, batch, tensor):
        clean_tensor = torch.where(
            self.get_non_pad_label_mask(batch, tensor),
            tensor, util.apply_gpu(torch.zeros(tensor.shape))
        )
        return clean_tensor
    
    def calculate_clean_metric(self, batch, raw_metric_tensor):
        non_pad_mask = self.get_non_pad_label_mask(batch, raw_metric_tensor)
        clean_metric_tensor = self.set_padded_to_zero(batch, raw_metric_tensor)
        
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
    
    def get_latent_distribution(self, latent_mean):
        return Normal(
            latent_mean,
            util.apply_gpu(torch.ones(latent_mean.shape)) * self.hparams.latent_sigma
        )
        
    def calculate_latent_kl(self, batch, latent_mean):
        latent_distribution = self.get_latent_distribution(self, latent_mean)
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
        pos_log_probs = F.log_softmax(self.decoder_pos(latent_sample), dim = -1)
        loss = self.encoder_loss(batch, pos_log_probs)
        return pos_log_probs, loss
        
    # Changed from forward.
    def __call__(self, batch):
        self.model.eval()
        current_language = batch['lang'][0]
        assert not any(list(filter(lambda example : example != current_language, batch['lang'])))
        
        loss = {} 
        hs = self.calculate_hidden_states(batch)
        latent_mean = self.encoder(hs)
        latent_sample = self.get_latent_distribution(latent_mean).rsample()
        predicted_hs = self.decoder_reconstruction(latent_sample)
        
        loss['latent_KL'] = self.calculate_latent_kl(batch, latent_mean)
        loss['MSE'] = self.calculate_masked_mse_loss(batch, predicted_hs, hs) 
        unlabeled_loss = self.hparams.nll_weight * loss['latent_KL'] + self.hparams.mse_weight * loss['MSE']
        
        # Labeled case,
        # but if training on English alone, then English should be treated as unsupervised.
        labeled_case = current_language == constant.SUPERVISED_LANGUAGE and len(self.hparams.trn_langs) > 1
        if labeled_case:
            with torch.no_grad():
                pos_log_probs, encoder_loss = self.calculate_encoder_outputs(batch, latent_sample)
            loss['total_loss'] = self.hparams.pos_nll_weight * loss['pos_nll'] + unlabeled_loss
        else:
            pos_log_probs, encoder_loss = self.calculate_encoder_outputs(batch, latent_sample)
            loss['total_loss'] = unlabeled_loss
            
        self.add_language_to_batch_output(loss, batch)
        return loss, pos_log_probs
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = Tagger.add_model_specific_args(parser)
        parser.add_argument("--latent_size", default=64, type=int)
        parser.add_argument("--latent_sigma", default=1, type=float)
        parser = Model.add_layer_stack_args(parser, 'encoder')
        parser = Model.add_layer_stack_args(parser, 'decoder_pos')
        parser = Model.add_layer_stack_args(parser, 'decoder_reconstruction')
        parser.add_argument("--pos_nll_weight", default=1, type=float)
        parser.add_argument("--latent_kl_weight", default=1, type=float)
        parser.add_argument("--mse_weight", default=1, type=float)
        return parser
        