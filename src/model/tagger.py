
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed forward to __call__.
# Added one layer MLP.
# Removed irrelevant code, such as outdated classes, imports, etc.
# and simplified to remove unused parameter choices.
# Added support for mBERT initialization for upper baselines.
# Changed to consider epoch metrics instead for equivalent checkpointing.
# Changed checkpointing metric for compatibility with wandb logging.
# Removed _metric.
# Added third return for forward for compatibility.
# Changed labels key for compatibility.

from copy import deepcopy
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial # added this from model/base.py

import util
from dataset import Dataset, UdPOS, LABEL_PAD_ID
from enumeration import Split, Task
from model.base import Model
from model.base_tagger import BaseTagger

# added below
import constant
from dataset import tagging
from torch.utils.data import DataLoader
# end added


class Tagger(BaseTagger):
    def __init__(self, hparams):
        super(Tagger, self).__init__(hparams)
        self._comparison_mode = 'max'
        self._selection_criterion = f'val_{constant.SUPERVISED_LANGUAGE}_pos_acc_epoch'

        if self.hparams.freeze_mbert:
            self.freeze_bert(self)
        # end additions

        self.id2label = UdPOS.get_labels()
        self.model_type = {
            'linear' : self.build_linear,
            'mlp' : self.build_mlp
        }
        self.classifier = self.model_type[self.hparams.base_pos_model_type](
            self.mbert_output_size, self.nb_labels,
            self.hparams.base_pos_hidden_size, self.hparams.base_pos_hidden_layers
        )
        
        # optimization loss added
        self.optimization_loss = 'pos_nll'
        self.metric_names = [
            self.optimization_loss,
            'pos_acc'
        ]
        self.setup_metrics()
        
    def extract_masked_representations(self, hs, raw_mask_indices):
        flat_mask_indices = raw_mask_indices.flatten()
        mask_indices = flat_mask_indices[flat_mask_indices != self.padding['mask_indices']]
        return torch.gather(hs, 1, mask_indices.reshape(-1, 1, 1).repeat(1, 1, hs.shape[-1])).squeeze()
        
    def __call__(self, batch):
        if self.hparams.freeze_mbert:
            self.encoder_mbert.eval()
        # Updated call arguments
        raw_hs = self.encode_sent(self.encoder_mbert, batch["sent"], batch["lang"])
        if self.hparams.masked:
            hs = self.extract_masked_representations(raw_hs, batch['mask_indices'])
            flat_padded_labels = batch['pos_labels'].flatten()
            labels = flat_padded_labels[flat_padded_labels != self.padding['pos_labels']]
            assert hs.shape[0] == labels.shape[0], f"hidden states: {hs.shape}, labels: {labels.shape}"
            assert len(hs.shape) == 2, hs.shape
            assert len(labels.shape) == 1, labels.shape
        else:
            hs, labels = raw_hs, batch['pos_labels']
        # end updates
        # removed use_crf
        logits = self.classifier(hs)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = self.calculate_encoder_loss(labels, log_probs)
        # Changed below to be compatible with later models' loss_dict and added assert.
        loss_dict = {self.optimization_loss : loss}
        self.add_language_to_batch_output(loss_dict, batch)
        return loss_dict, { 'pos' : (log_probs, labels) }, None
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = Model.add_model_specific_args(parser)
        parser = Model.add_layer_stack_args(parser, 'base_pos')
        return parser
        

