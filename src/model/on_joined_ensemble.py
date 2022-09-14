
from model.tagger import Tagger
from dataset.uniform_view import UniformViewDataset
import util

import torch

class OnJoinedEnsemble(Tagger):
    
    def __init__(self, hparams):
        super(OnJoinedEnsemble, self).__init__(hparams)
        self._data_class = UniformViewDataset
        
    def get_self_training_args(self):
        return {
            'view_checkpoint_1' : self.hparams.joined_view_checkpoint,
            'view_checkpoint_2' : self.hparams.joined_view_checkpoint,
            'is_masked_view_1' : self.hparams.is_masked_view_1,
            'is_masked_view_2' : self.hparams.is_masked_view_2,
        }
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = Tagger.add_model_specific_args(parser)
        parser.add_argument("--joined_view_checkpoint", default="", type=str)
        parser.add_argument("--is_masked_view_1", default=False, type=util.str2bool)
        parser.add_argument("--is_masked_view_2", default=False, type=util.str2bool)
        return parser
        
    