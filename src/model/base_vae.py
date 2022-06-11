
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
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
import numpy as np

from dataset import LABEL_PAD_ID
from dataset import tagging
from model.tagger import Tagger
from enumeration import Split
import util
import metric
import constant

class BaseVAE(Tagger):
    
    def __init__(self, hparams):
        super(BaseVAE, self).__init__(hparams)
        encoder = Tagger.load_from_checkpoint(self.hparams.encoder_checkpoint)
        self.freeze_bert(encoder)
        # Overwrite base and tagger attributes so that encode_sent will function correctly
        self.model = encoder.model
        self.classifier = encoder.classifier
        self.decoder_lstm = torch.nn.LSTM(
                input_size = len(constant.UD_POS_LABELS),
                hidden_size = self.hidden_size, # Encoder input size
                num_layers = self.hparams.decoder_number_of_layers,
                batch_first = True,
                bidirectional = True
        )
        self.decoder_linear = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fixed_prior = self.get_smoothed_english_prior()
        self._selection_criterion = 'decoder_loss'
        
    def freeze_bert(self, encoder):
        # Adapted from model/base.py by taking the logic to freeze up to and including a certain layer
        # Doesn't freeze the pooler, but encode_sent excludes pooler correctly.
        encoder.freeze_embeddings()
        for index in range(encoder.model.config.num_hidden_layers + 1):
            encoder.freeze_layer(index)
        # end adapted
    
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
        return repeated_labels != metric.LABEL_PAD_ID        
        
    def set_padded_to_zero(self, batch, tensor):
        clean_tensor = torch.where(
            self.get_non_pad_label_mask(batch, tensor),
            tensor, util.apply_gpu(torch.zeros(tensor.shape))
        )
        return clean_tensor
        
    def calculate_decoder(self, pi_t):
        # "Reshape" the bidirectional concatenation
        mu_t_raw, _ = self.decoder_lstm(pi_t)
        mu_t = self.decoder_linear(mu_t_raw)
        return mu_t
        
    def masked_mse_loss(self, batch, padded_mu_t, padded_hs):
        clean_mu_t = self.set_padded_to_zero(batch, padded_mu_t)
        clean_hs = self.set_padded_to_zero(batch, padded_hs)
        clean_difference_sum = torch.sum(torch.pow(clean_mu_t - clean_hs, 2))
        assert clean_hs.shape == clean_mu_t.shape, f'hs: {clean_hs.shape}, mu_t: {clean_mu_t.shape}'
        non_pad_mask = self.get_non_pad_label_mask(batch, clean_hs)
        clean_mse = clean_difference_sum / torch.sum(non_pad_mask)
        return clean_mse
        
    def calculate_agnostic_loss(self, batch, mu_t, hs):
         # Calculate losses
        loss = {}
        loss['MSE'] = self.masked_mse_loss(batch, mu_t, hs)
        return loss
        
    def forward(self, batch):
        # Padded true_pi_t will be all 0.
        assert len(batch['labels'].shape) == 2
        true_pi_t = F.one_hot(
            batch['labels'] + 1,
            num_classes = len(constant.UD_POS_LABELS) + 1
        )[:, :, 1:].float()
        
        hs = self.calculate_hidden_states(batch)
        mu_t = self.calculate_decoder(true_pi_t)
        loss = self.calculate_agnostic_loss(batch, mu_t, hs)
        
        loss['decoder_loss'] = loss['MSE']
        return loss
        
    # end forward-related methods
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--temperature", default=1, type=float)
        parser.add_argument("--decoder_number_of_layers", default=1, type=int)
        return parser
    
    def get_uniform_prior(self):
        number_of_labels = len(constant.UD_POS_LABELS)
        return torch.ones(number_of_labels) / number_of_labels
        
    def get_english_prior(self):
        counts = self.get_label_counts('English', Split.train)
        return counts / torch.sum(counts)
        
    def get_smoothed_english_prior(self):
        threshold = 0.001
        prior = self.get_english_prior()
        raw_smoothed_prior = torch.clamp(prior, min = threshold)

        assert torch.all(raw_smoothed_prior >= threshold), raw_smoothed_prior
        return raw_smoothed_prior / torch.sum(raw_smoothed_prior)
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        loss = loss_dict['decoder_loss']
        self.log("loss", loss)
        
        try:
            return loss
        except:
            import pdb
            pdb.set_trace()
    
    def evaluation_step_helper(self, batch, prefix):
        return self.forward(batch)
    