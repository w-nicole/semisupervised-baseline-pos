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
import numpy as np

from metric import LABEL_PAD_ID
from dataset import tagging
from model.tagger import Tagger
from enumeration import Split
import util
import constant

class BaseVAE(Tagger):
    
    def __init__(self, hparams):
        super(BaseVAE, self).__init__(hparams)
        if self.hparams.encoder_checkpoint:
            encoder = Tagger.load_from_checkpoint(self.hparams.encoder_checkpoint)
        else:
            encoder = Tagger(self.hparams)
        self.freeze_bert(encoder)
        # Overwrite base and tagger attributes so that encode_sent will function correctly
        self.model = encoder.model
        self.classifier = encoder.classifier
        self.decoder_lstm = torch.nn.LSTM(
                input_size = len(constant.UD_POS_LABELS) + self.hparams.auxiliary_size,
                hidden_size = self.hidden_size, # Encoder input size
                num_layers = self.hparams.decoder_number_of_layers,
                bidirectional = True
        )
        self.use_auxiliary = self.hparams.auxiliary_size > 0
        if self.use_auxiliary:
            if self.hparams.auxiliary_hidden_layers >= 0:
                self.auxiliary_mu = self.build_mlp(
                    self.hidden_size,
                    self.hparams.auxiliary_size,
                    self.hparams.auxiliary_hidden_size,
                    self.hparams.auxiliary_hidden_layers,
                    self.hparams.auxiliary_nonlinear_first
                )
                self.auxiliary_sigma = self.build_mlp(
                    self.hidden_size,
                    self.hparams.auxiliary_size,
                    self.hparams.auxiliary_hidden_size,
                    self.hparams.auxiliary_hidden_layers,
                    self.hparams.auxiliary_nonlinear_first
                )
            else:
                self.auxiliary_mu = torch.nn.Linear(self.hidden_size, self.hparams.auxiliary_size)
                self.auxiliary_sigma = torch.nn.Linear(self.hidden_size, self.hparams.auxiliary_size)
            
        #self.decoder_linear = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        print('Correct back in BaseVAE.__init__: currently is a single layer')
        self.decoder_linear = torch.nn.Linear(self.nb_labels, self.hidden_size)
        self._selection_criterion = f'val_{self.hparams.trn_langs[0]}_decoder_loss'
        self._comparison_mode = 'min'
        self.optimization_loss = 'decoder_loss'
    
    # Below forward-related methods:
    # Shijie Wu's code, but with decoder logic added and irrelevant options removed,
    # and variables renamed for notation consistency.
    
    def calculate_hidden_states(self, batch):
        batch = self.preprocess_batch(batch)
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
        
    def calculate_decoder(self, pi_t):
        # "Reshape" the bidirectional concatenation
        #mu_t_raw, _ = self.decoder_lstm(pi_t)
        #mu_t = self.decoder_linear(mu_t_raw)
        mu_t = self.decoder_linear(F.softmax(pi_t, dim=-1))
        return mu_t
        
    def calculate_decoder_loss(self, batch, hs, pi_t):
        loss = {}
        if self.use_auxiliary:
            auxiliary_mu_t = self.auxiliary_mu(hs)
            auxiliary_sigma_t = torch.exp(self.auxiliary_sigma(hs))

            auxiliary_distribution = Normal(auxiliary_mu_t, auxiliary_sigma_t)
            auxiliary = auxiliary_distribution.rsample()
            normal_prior = Normal(
                util.apply_gpu(torch.zeros(auxiliary.shape)),
                util.apply_gpu(torch.ones(auxiliary.shape))
            )
            loss['auxiliary_KL'] = torch.distributions.kl.kl_divergence(auxiliary_distribution, normal_prior).sum(dim=-1).mean()
            decoder_input = torch.cat([pi_t, auxiliary], dim = -1)
        else:
            decoder_input = pi_t

        mu_t = self.calculate_decoder(decoder_input)

        loss['MSE'] = self.masked_mse_loss(batch, mu_t, hs)
        auxiliary_loss = self.hparams.auxiliary_kl_weight * loss['auxiliary_KL'] if self.use_auxiliary else 0
        loss['decoder_loss'] = loss['MSE'] + auxiliary_loss

        return loss
        
    def masked_mse_loss(self, batch, padded_mu_t, padded_hs):
        clean_mu_t = self.set_padded_to_zero(batch, padded_mu_t)
        clean_hs = self.set_padded_to_zero(batch, padded_hs)
        
        clean_difference_sum = torch.sum(torch.pow(clean_mu_t - clean_hs, 2))
        assert clean_hs.shape == clean_mu_t.shape, f'hs: {clean_hs.shape}, mu_t: {clean_mu_t.shape}'
        non_pad_mask = self.get_non_pad_label_mask(batch, clean_hs)
        # Adjust scale to NOT divide out the hidden size representation.
        clean_mse = clean_difference_sum / torch.sum(non_pad_mask) * padded_hs.shape[-1]
        return clean_mse
        
    # Changed from forward.
    def __call__(self, batch):
        self.model.eval()
        # Padded true_pi_t will be all 0.
        assert len(batch['labels'].shape) == 2
        true_pi_t = F.one_hot(
            batch['labels'] + 1,
            num_classes = len(constant.UD_POS_LABELS) + 1
        )[:, :, 1:].float()
        
        hs = self.calculate_hidden_states(batch)
        loss = self.calculate_decoder_loss(batch, hs, true_pi_t)
        loss['decoder_loss'] = loss['MSE']

        self.add_language_to_batch_output(loss, batch)
        return loss, None
        
    # end forward-related methods
    
    # Renamed variables, function, direct return of loss_dict, no self.log for loss
    # Updated assert message and metrics indexing
    def step_helper(self, batch, prefix):
        loss_dict, _ = self.__call__(batch)
        return loss_dict
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--auxiliary_kl_weight", default=1, type=int)
        # auxiliary_hidden_layers = -1 indicates no hidden layers (single linear layer).
        parser.add_argument("--auxiliary_hidden_layers", default=-1, type=int)
        parser.add_argument("--auxiliary_hidden_size", default=0, type=int)
        # auxiliary_size = 0 indicates no auxiliary vector.
        parser.add_argument("--auxiliary_size", default=0, type=int)
        parser.add_argument("--auxiliary_nonlinear_first", default=False, type=util.str2bool)
        parser.add_argument("--temperature", default=1, type=float)
        parser.add_argument("--decoder_number_of_layers", default=1, type=int)
        return parser
