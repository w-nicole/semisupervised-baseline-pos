
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
        self._selection_criterion = f'val_{self.target_language}_acc_epoch'

        self.is_frozen_mbert = self.hparams.mbert_checkpoint or self.hparams.freeze_mbert
        assert not ((not self.hparams.freeze_mbert) and self.hparams.mbert_checkpoint),\
            "Mutually exclusive. mbert_checkpoint always automatically frozen."
            
        # Reinitialize mBERT alone if given a checkpoint.
        if self.hparams.mbert_checkpoint:
            encoder = Tagger.load_from_checkpoint(self.hparams.mbert_checkpoint)
            self.freeze_bert(encoder)
            self.model = encoder.model
            self.concat_all_hidden_states = encoder.concat_all_hidden_states
        
        if self.hparams.freeze_mbert:
            self.freeze_bert(self)
        # end additions

        self.id2label = UdPOS.get_labels()
        self.model_type = {
            'linear' : self.build_linear,
            'mlp' : self.build_mlp
        }
        self.classifier = self.model_type[self.hparams.pretrained_model_type](
            self.mbert_output_size, self.nb_labels,
            self.hparams.pretrained_hidden_size, self.hparams.pretrained_hidden_layers
        )
        
        # optimization loss added
        self.optimization_loss = 'pos_nll'
        self.metric_names = [
            self.optimization_loss,
            'acc'
        ]
        self.setup_metrics()
    
    def __call__(self, batch):
        if self.is_frozen_mbert:
            self.model.eval()
        # Updated call arguments
        hs = self.encode_sent(self.model, batch["sent"], batch["start_indices"], batch["end_indices"], batch["lang"])
        # end updates
        # removed use_crf
        logits = self.classifier(hs)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(
            log_probs.view(-1, self.nb_labels),
            batch["labels"].view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        # Changed below to be compatible with later models' loss_dict and added assert.
        loss_dict = {self.optimization_loss : loss}
        self.add_language_to_batch_output(loss_dict, batch)
        return loss_dict, log_probs
        
    @classmethod
    def add_model_specific_args(cls, parser):
        # Added these arguments, removed crf argument
        # -1 indicates a linear layer alone (no input layer).
        parser.add_argument("--freeze_mbert", default=False, type=util.str2bool)
        parser.add_argument("--mbert_checkpoint", default="", type=str)
        parser = Model.add_model_specific_args(parser)
        parser = Model.add_layer_stack_args(parser, 'pretrained')
        parser.add_argument('--pretrained_model_type', default='linear', type=str)
        return parser
        

