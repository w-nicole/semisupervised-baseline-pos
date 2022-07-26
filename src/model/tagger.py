
# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Added averaging behavior,
#   and changed "forward" accordingly.
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
from metric import LABEL_PAD_ID # changed this from dataset import
from dataset import Dataset, UdPOS
from enumeration import Split, Task
from model.base import Model
from model.base_tagger import BaseTagger

# added below
import constant
from dataset import tagging
from torch.utils.data import DataLoader
# end added

# Below: added imports
from dataset import collate
# end imports


class Tagger(BaseTagger):
    def __init__(self, hparams):
        super(Tagger, self).__init__(hparams)
        self._comparison_mode = 'max'
        self._selection_criterion = f'val_{self.target_language}_pos_acc_epoch'

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
    
    def __call__(self, batch):
        if self.hparams.freeze_mbert:
            self.encoder_mbert.eval()
        # Updated call arguments
        hs = self.encode_sent(self.encoder_mbert, batch["sent"], batch["averaging_indices"], batch["lang"])
        # end updates
        # removed use_crf
        logits = self.classifier(hs)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(
            log_probs.view(-1, self.nb_labels),
            batch["pos_labels"].view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        # Changed below to be compatible with later models' loss_dict and added assert.
        loss_dict = {self.optimization_loss : loss}
        self.add_language_to_batch_output(loss_dict, batch)
        return loss_dict, { 'pos' : log_probs }, None
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = Model.add_model_specific_args(parser)
        parser = Model.add_layer_stack_args(parser, 'base_pos')
        return parser
        

