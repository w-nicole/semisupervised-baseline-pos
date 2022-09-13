
from model.tagger import Tagger
from dataset.single import SingleDataset

import torch

class JoinedEnsemble(Tagger):
    
    def __init__(self, hparams):
        super(JoinedEnsemble, self).__init__(hparams)
        assert self.hparams.unraveled_predictions and self.hparams.mask_probability > 0,\
            f"{self.hparams.unraveled_predictions}, {self.hparams.mask_probability}"
        assert self.hparams.double_pass
        self._data_class = SingleDataset
        
    def __call__(self, batch):
        common_args = { 'pos_labels' : batch['pos_labels'], 'lang' : batch['lang'] }
        args_1 = { 'sent' : batch['sent_1'] }
        args_2 = { 'sent' : batch['sent_2'] }
        args_1.update(common_args)
        args_2.update(common_args)
        
        loss_dict_1, log_probs_dict_1, _ = super(JoinedEnsemble, self).__call__(args_1)
        loss_dict_2, log_probs_dict_2, _ = super(JoinedEnsemble, self).__call__(args_2)
        
        loss_key = self.optimization_loss
        loss_dict = { loss_key : (loss_dict_1[loss_key] + loss_dict_2[loss_key]) / 2 }
        
        concat_log_probs = torch.cat([
                log_probs_dict_1['pos'][0], log_probs_dict_2['pos'][0]
            ], axis = 0)
        if not torch.all(log_probs_dict_1['pos'][1] == log_probs_dict_2['pos'][1])\
            and torch.all(batch['pos_labels'] == log_probs_dict_1['pos'][1]):
                import pdb; pdb.set_trace()
        concat_labels = torch.cat([
                batch['pos_labels'], batch['pos_labels']
            ], axis = 0)
        log_probs_dict = { 'pos' : (concat_log_probs, concat_labels) } 
        
        return loss_dict, log_probs_dict, None
        
