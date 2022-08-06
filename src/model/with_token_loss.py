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
from transformers import BertForMaskedLM

import util
from metric import LABEL_PAD_ID
from model.latent_base import LatentBase

class WithTokenLoss(LatentBase):
    
    def __init__(self, hparams):
        super(WithTokenLoss, self).__init__(hparams)
        assert not self.hparams.concat_all_hidden_states, "Not supported for WithTokenLoss."
        self.token_classifier = BertForMaskedLM.from_pretrained(self.hparams.pretrain).cls
        util.freeze(self.token_classifier)
        self.metric_names.extend([
            'token_acc',
            'token_nll',
            'mask_kl'
        ])
        self.setup_metrics()
        
    def calculate_token_nll_loss(self, batch, log_probs):
        encoder_loss = F.nll_loss(
            log_probs.view(-1, log_probs.shape[-1]),
            batch["token_labels"].view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        return encoder_loss
        
    def calculate_kl(self, batch, log_p, log_q):
        pre_sum = torch.exp(log_p) * (log_p - log_q)
        assert pre_sum.shape == log_p.shape, f'pre_sum: {pre_sum.shape}, log_p: {log_p.shape}'
        kl_divergence_mean = self.calculate_clean_metric(batch['token_labels'], pre_sum)
        return kl_divergence_mean
        
    def __call__(self, batch):
        
        self.token_classifier.eval()
        loss, pos_log_probs, model_outputs = super(WithTokenLoss, self).__call__(batch)
        
        token_logits = self.token_classifier(model_outputs['predicted_hs'])
        log_probs = F.log_softmax(token_logits, dim = -1)
        
        with torch.no_grad():
            true_token_logits = self.token_classifier(model_outputs['target_hs'])
            true_log_probs = F.log_softmax(true_token_logits, dim = -1)

        loss['token_nll'] = self.calculate_token_nll_loss(batch, log_probs)
        loss['mask_kl'] = self.calculate_kl(batch, true_log_probs, log_probs)
        log_probs_dict = {'token' : log_probs}
        log_probs_dict.update(pos_log_probs)
        
        loss['total_loss'] += self.hparams.token_nll_weight * loss['token_nll'] + self.hparams.mask_kl_weight * loss['mask_kl']
        return loss, log_probs_dict, model_outputs
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = LatentBase.add_model_specific_args(parser)
        parser.add_argument('--token_nll_weight', default=1, type=float)
        parser.add_argument('--mask_kl_weight', default=1, type=float)
        parser.add_argument('--debug_model_all_eval', default=False, type=util.str2bool)
        return parser
        