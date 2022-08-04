
from model.latent_base import LatentBase
import util

class LatentSpace(LatentBase):
    def __init__(self, hparams):
        super(LatentSpace, self).__init__(hparams)
        self.decoder_pos = None # for backprop speed
        self._selection_criterion = 'val_all_MSE_epoch'
        self._comparison_mode = 'min'
        self.optimization_loss = 'MSE'
        self.metric_names = [
            'MSE'
        ]
        self.setup_metrics()
        
    def __call__(self, batch):
        loss = {} 
        encoder_hs, latent, target_hs = self.calculate_intermediates(batch)
        predicted_hs = self.decoder_reconstruction(*self.get_decoder_args(self.decoder_reconstruction, batch, latent))
        loss['MSE'] = self.calculate_masked_mse_loss(batch, predicted_hs, target_hs)
        self.add_language_to_batch_output(loss, batch)
        
        # empty dict will correctly skip the accuracy metric in this case
        return loss, { }, { 'predicted_hs' : predicted_hs }