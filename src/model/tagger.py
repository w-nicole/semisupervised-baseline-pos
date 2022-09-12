
# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

from copy import deepcopy
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial # added this from model/base.py

import util
from dataset.tagging import UdPOS
from constant import LABEL_PAD_ID
from enumeration import Split, Task
from model.base import Model
from model.base_tagger import BaseTagger

import constant
from torch.utils.data import DataLoader


class Tagger(BaseTagger):
    def __init__(self, hparams):
        super(Tagger, self).__init__(hparams)
        self._comparison_mode = 'max'
        self._selection_criterion = f'val_{self.hparams.target_language}_pos_acc_epoch'

        if self.hparams.freeze_mbert:
            self.freeze_bert(self)

        self.id2label = UdPOS.get_labels()
        self.model_type = {
            'linear' : self.build_linear,
            'mlp' : self.build_mlp
        }
        self.classifier = self.model_type[self.hparams.base_pos_model_type](
            self.mbert_output_size, self.nb_labels,
            self.hparams.base_pos_hidden_size, self.hparams.base_pos_hidden_layers
        )
        
        self.optimization_loss = 'pos_nll'
        self.metric_names = [
            self.optimization_loss,
            'pos_acc'
        ]
        self.setup_metrics()
        
    def __call__(self, batch):
        if self.hparams.freeze_mbert:
            self.encoder_mbert.eval()
        hs = self.encode_sent(self.encoder_mbert, batch["sent"], batch["lang"])
        labels = batch['pos_labels']
        logits = self.classifier(hs)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = self.calculate_encoder_loss(labels, log_probs)
        loss_dict = {self.optimization_loss : loss}
        self.add_language_to_batch_output(loss_dict, batch)
        return loss_dict, { 'pos' : (log_probs, labels) }, None
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = Model.add_model_specific_args(parser)
        parser = Model.add_layer_stack_args(parser, 'base_pos')
        return parser
        

