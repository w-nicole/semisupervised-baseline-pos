from model.tagger import Tagger
from dataset.on_split_ensemble import OnSplitEnsembleDataset

import util

class OnSplitEnsemble(Tagger):
    
    def __init__(self, hparams):
        super(OnSplitEnsemble, self).__init__(hparams)
        self._data_class = OnSplitEnsembleDataset
        
    def get_self_training_args(self):
        return {
            'pseudolabel_checkpoint' : self.hparams.pseudolabel_checkpoint,
            'view_checkpoint_1' : self.hparams.view_checkpoint_1,
            'view_checkpoint_2' : self.hparams.view_checkpoint_2,
            'view_mask_probability_1' : self.hparams.view_mask_probability_1,
            'view_mask_probability_2' : self.hparams.view_mask_probability_2,
        }
        
    @classmethod
    def add_model_specific_args(cls, parser):
        # self-training arguments
        parser = Tagger.add_model_specific_args(parser)
        parser.add_argument("--pseudolabel_checkpoint", default="", type=str)
        parser.add_argument("--view_checkpoint_1", default="", type=str)
        parser.add_argument("--view_checkpoint_2", default="", type=str)
        parser.add_argument("--view_mask_probability_1", default=0, type=float)
        parser.add_argument("--view_mask_probability_2", default=0, type=float)
        return parser