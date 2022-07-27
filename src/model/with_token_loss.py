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
from model.latent_base import LatentBase

import util
from transformers import BertForMaskedLM

import util
from metric import LABEL_PAD_ID

class WithTokenLoss(LatentBase):
    
    def __init__(self, hparams):
        super(WithTokenLoss, self).__init__(hparams)
        assert not self.hparams.concat_all_hidden_states, "Not supported for WithTokenLoss."
        self.token_classifier = BertForMaskedLM.from_pretrained(self.hparams.pretrain).cls
        util.freeze(self.token_classifier)
        self.metric_names.extend([
            'token_acc',
            'token_nll'
        ])
        self.setup_metrics()
        
    def calculate_token_loss(self, batch, log_probs):
        encoder_loss = F.nll_loss(
            log_probs.view(-1, log_probs.shape[-1]),
            batch["token_labels"].view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        return encoder_loss
        
    def __call__(self, batch):
        
        loss, pos_log_probs, model_outputs = super(WithTokenLoss, self).__call__(batch)
        
        token_logits = self.token_classifier(model_outputs['predicted_hs'])
        log_probs = F.log_softmax(token_logits, dim = -1)
        
        loss['token_nll'] = self.calculate_token_loss(batch, log_probs)
        log_probs_dict = {'token' : log_probs}
        log_probs_dict.update(pos_log_probs)
        
        return loss, log_probs_dict, model_outputs