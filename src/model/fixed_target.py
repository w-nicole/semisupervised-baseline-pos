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
from model.latent_base import LatentBase

class FixedTarget(LatentBase):
    def __init__(self, hparams):
        super(FixedTarget, self).__init__(hparams)
        target_tagger = self.load_tagger(self.hparams.target_mbert_checkpoint)
        self.freeze_bert(target_tagger)
        self.target_mbert = target_tagger.model
        
        # Check concat status
        assert target_tagger.concat_all_hidden_states == self.concat_all_hidden_states,\
            f'target: {target_tagger.concat_all_hidden_states}, encoder: {self.concat_all_hidden_states}'
        
    def calculate_target_hs(self, batch, predicted_hs):
        with torch.no_grad():
            self.target_mbert.eval()
            target_hs = self.calculate_hidden_states(self.target_mbert, batch)
        return target_hs
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = LatentBase.add_model_specific_args(parser)
        parser.add_argument('--target_mbert_checkpoint', default='', type=str)
        return parser