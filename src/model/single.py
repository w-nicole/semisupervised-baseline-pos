
from model.tagger import Tagger
from dataset.single import SingleDataset

class Single(Tagger):
    
    def __init__(self, hparams):
        super(Single, self).__init__(hparams)
        self._data_class = SingleDataset
