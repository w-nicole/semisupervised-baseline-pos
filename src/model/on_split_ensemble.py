
from model.tagger import Tagger
from dataset.uniform_view import UniformViewDataset

import util

class OnSplitEnsemble(Tagger):
    
    def __init__(self, hparams):
        super(OnSplitEnsemble, self).__init__(hparams)
        self._data_class = UniformViewDataset
        OnSplitEnsemble.check_self_training_args(hparams)
        
    @classmethod
    def check_self_training_args(cls, hparams):
        is_self_training = hparams.view_checkpoint_1 and hparams.view_checkpoint_2
        is_regular = not (hparams.view_checkpoint_1 or hparams.view_checkpoint_2)
        assert is_self_training or is_regular, f"{hparams.view_checkpoint_1}, {hparams.view_checkpoint_2}"
        if is_regular:
            is_unspecified = lambda item : bool(item)
            assert not any(map(is_unspecified, [
                    hparams.view_checkpoint_1,
                    hparams.view_checkpoint_2,
                    hparams.is_masked_view_1,
                    hparams.is_masked_view_2,
                    hparams.use_subset_complement,
                ]))
        is_masked_paths = {}
        # Below: temp checks for masked argument specification.
        # Below don't necessarily need to hold, but in the present setup, they do.
        is_unmasked_list = []
        for path, specified_masked in zip(
                [hparams.view_checkpoint_1, hparams.view_checkpoint_2],
                [hparams.is_masked_view_1, hparams.is_masked_view_2]
            ):
            is_unmasked = 'unmasked' in path
            if not is_unmasked: assert 'masked' in path, path
            assert (not is_unmasked) == specified_masked,\
                f'path: {path}, specified: {specified_masked}, actual: {is_unmasked}'
            is_unmasked_list.append(is_unmasked)
        # Below: this is only designed for mixed and pure_unmasked for now
        assert (is_unmasked_list[0] ^ is_unmasked_list[1]) or all(is_unmasked_list), is_unmasked_list
        
    def get_self_training_args(self):
        return {
            'view_checkpoint_1' : self.hparams.view_checkpoint_1,
            'view_checkpoint_2' : self.hparams.view_checkpoint_2,
            'is_masked_view_1' : self.hparams.is_masked_view_1,
            'is_masked_view_2' : self.hparams.is_masked_view_2,
        }
        
    @classmethod
    def add_model_specific_args(cls, parser):
        # self-training arguments
        parser = Tagger.add_model_specific_args(parser)
        parser.add_argument("--view_checkpoint_1", default="", type=str)
        parser.add_argument("--view_checkpoint_2", default="", type=str)
        # TODO: replace below with a hparam read
        parser.add_argument("--is_masked_view_1", default=False, type=util.str2bool)
        parser.add_argument("--is_masked_view_2", default=False, type=util.str2bool)
        return parser

