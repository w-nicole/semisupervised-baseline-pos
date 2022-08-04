
from model.latent_space import LatentSpace
from model.base_tagger import BaseTagger

import torch
import torch.nn.functional as F
import constant
import util

class LatentProbe(BaseTagger):
    def __init__(self, hparams):
        super(LatentProbe, self).__init__(hparams)
        self.encoder = LatentSpace.load_from_checkpoint(self.hparams.latent_space_checkpoint)
        util.freeze(self.encoder)
        self.classifier = torch.nn.Linear(self.encoder.hparams.latent_size, self.nb_labels)
        self._selection_criterion = f'val_{constant.SUPERVISED_LANGUAGE}_pos_acc_epoch'
        self._comparison_mode = 'max'
        self.optimization_loss = 'pos_nll'
        self.metric_names = [
            'pos_nll',
            'pos_acc'
        ]
        self.setup_metrics()
        
    def __call__(self, batch):
        loss = {} 
        _, latent = self.encoder.calculate_encoder_intermediates(batch)
        logits = self.classifier(latent)
        log_probs = F.log_softmax(logits, dim = -1)
        loss['pos_nll'] = self.calculate_encoder_loss(batch, log_probs)
        self.add_language_to_batch_output(loss, batch)
        return loss, { 'pos' : log_probs }, None
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser = BaseTagger.add_model_specific_args(parser)
        parser.add_argument('--latent_space_checkpoint', default='', type=str)
        return parser