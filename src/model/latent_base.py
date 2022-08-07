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


class LatentBase(BaseTagger):
    
    def __init__(self, hparams):
        super(LatentBase, self).__init__(hparams)
        self.target_mbert = self.build_model(self.hparams.pretrain)
        util.freeze(self.target_mbert) 

        encoder_args = (
            self.mbert_output_size, self.hparams.latent_size,
            self.hparams.encoder_hidden_size, self.hparams.encoder_hidden_layers
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
            'mlp' : self.build_mlp,
            'linear' : self.build_linear
        }
        if not self.hparams.debug_fix_identity:
            self.encoder = self.model_type[self.hparams.encoder_model_type](*encoder_args)
        else:
            self.encoder = torch.nn.Linear(768, 768)
            self.encoder.weight = torch.nn.parameter.Parameter(data=torch.eye(768), requires_grad = False)
            self.encoder.bias = torch.nn.parameter.Parameter(data=torch.zeros(768,), requires_grad = False)
            util.freeze(self.encoder)
        self.decoder_pos = self.model_type[self.hparams.pos_model_type](*pos_model_args)
        self.decoder_reconstruction = self.model_type[self.hparams.reconstruction_model_type](*reconstruction_model_args)
        self.optimization_loss = 'total_loss'
        self._selection_criterion = f'val_{constant.SUPERVISED_LANGUAGE}_pos_acc_epoch'
        self._comparison_mode = 'max'
        self.metric_names = [
            'supervised_pos_nll',
            'total_loss',
            'MSE',
            'pos_acc'
        ]
        self.setup_metrics()
        
    def calculate_hidden_states(self, mbert, batch):
        # Updated call arguments
        hs = self.encode_sent(mbert, batch["sent"], batch["lang"])
        return hs
    
    def get_non_pad_label_mask(self, labels, tensor):
        assert len(labels.shape), labels.shape
        repeated_labels = labels.unsqueeze(2).repeat(1, 1, tensor.shape[-1])
        return util.apply_gpu(repeated_labels != LABEL_PAD_ID)   
        
    def set_padded_to_zero(self, labels, tensor):
        clean_tensor = torch.where(
            self.get_non_pad_label_mask(labels, tensor),
            tensor, util.apply_gpu(torch.zeros(tensor.shape))
        )
        return clean_tensor
    
    def calculate_clean_metric(self, labels, raw_metric_tensor):
        non_pad_mask = self.get_non_pad_label_mask(labels, raw_metric_tensor)
        clean_metric_tensor = self.set_padded_to_zero(labels, raw_metric_tensor)
        
        # for KL divergence, but MSE doesn't trigger false positive
        metric_per_position = torch.sum(clean_metric_tensor, axis = -1)
        if not torch.all(metric_per_position >= 0):
            import pdb; pdb.set_trace()
        
        # Adjust scale to NOT divide out the hidden size representation.
        clean_average = torch.sum(clean_metric_tensor) / torch.sum(non_pad_mask) * raw_metric_tensor.shape[-1]
        return clean_average
        
    def calculate_masked_mse_loss(self, batch, padded_mu_t, padded_hs):
        raw_metric_tensor = torch.pow(padded_mu_t - padded_hs, 2)
        clean_mse = self.calculate_clean_metric(batch['pos_labels'], raw_metric_tensor)
        return clean_mse
    
    def calculate_encoder_outputs(self, pos_labels, latent):
        decoder_output = self.decoder_pos(latent)
        pos_log_probs = F.log_softmax(decoder_output, dim = -1)
        loss = self.calculate_encoder_loss(pos_labels, pos_log_probs)
        return pos_log_probs, loss
        
    def calculate_encoder_intermediates(self, batch):
        if self.hparams.freeze_mbert:
            self.encoder_mbert.eval()
        encoder_hs = self.calculate_hidden_states(self.encoder_mbert, batch)
        latent = self.encoder(encoder_hs)
        return encoder_hs, latent
        
    def calculate_intermediates(self, batch):
        encoder_hs, latent = self.calculate_encoder_intermediates(batch)
        target_hs = self.calculate_target_hs(batch)
        return encoder_hs, latent, target_hs
      
    def calculate_target_hs(self, batch):
        with torch.no_grad():
            self.target_mbert.eval()
            target_hs = self.calculate_hidden_states(self.target_mbert, batch)
        return target_hs
        
    def mask_tensor_to_supervised(self, tensor, is_supervised):
        assert len(is_supervised.shape) == 1, is_supervised.shape
        if len(tensor.shape) == 3:
            unsqueezed = is_supervised.reshape(is_supervised.shape[0], 1, 1)
        elif len(tensor.shape) == 2:
            unsqueezed = is_supervised.reshape(is_supervised.shape[0], 1)
        else:
            assert False, f"{tensor.shape} not supported"
        supervised_mask = unsqueezed.repeat(*((1,) + tensor.shape[1:]))
        
        assert torch.all((is_supervised == 0) | (is_supervised == 1)), torch.unique(is_supervised)
        supervised_tensor = torch.where(
            (supervised_mask.int() == 1), tensor,
            LABEL_PAD_ID * util.apply_gpu(torch.ones(tensor.shape))
        )
        return supervised_tensor
        
    def __call__(self, batch):
        current_language = batch['lang'][0]
        assert not any(list(filter(lambda example : example != current_language, batch['lang'])))
        
        loss = {} 
        encoder_hs, latent, target_hs = self.calculate_intermediates(batch)
        predicted_hs = self.decoder_reconstruction(latent)
        loss['MSE'] = self.calculate_masked_mse_loss(batch, predicted_hs, target_hs)
        unlabeled_loss = self.hparams.mse_weight * loss['MSE']
        
        supervised_latent = self.mask_tensor_to_supervised(latent, batch['is_supervised'])

        supervised_labels = self.mask_tensor_to_supervised(batch['pos_labels'], batch['is_supervised'])
        _, supervised_encoder_loss = self.calculate_encoder_outputs(batch['pos_labels'], supervised_latent)
        loss['supervised_pos_nll'] = encoder_loss
        
        loss['total_loss'] = self.hparams.pos_nll_weight * supervised_encoder_loss + unlabeled_loss
        with torch.no_grad():
            pos_log_probs, _ = self.calculate_encoder_outputs(batch['pos_labels'], latent)
        
        self.add_language_to_batch_output(loss, batch)
        return loss, { 'pos_log_probs' : pos_log_probs }, { 'target_hs' : target_hs, 'predicted_hs' : predicted_hs } 
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = Tagger.add_model_specific_args(parser)
        parser.add_argument("--latent_size", default=64, type=int)
        parser = Model.add_layer_stack_args(parser, 'encoder')
        parser = Model.add_layer_stack_args(parser, 'pos')
        parser = Model.add_layer_stack_args(parser, 'reconstruction')
        parser.add_argument("--pos_nll_weight", default=1, type=float)
        parser.add_argument("--mse_weight", default=1, type=float)
        parser.add_argument("--debug_fix_identity", default=False, type=util.str2bool)
        return parser
        