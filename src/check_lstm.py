
import os
import torch

import yaml
from argparse import Namespace
import torch
import torch.nn.functional as F
import numpy as np
from metric import LABEL_PAD_ID

from model import LatentToPOS
import pytorch_lightning as pl

import explore_predict

def prep_fresh_load_model(hparams):
    model = LatentToPOS(hparams).cuda()
    model.encoder_mbert = model.encoder_mbert.cuda()
    model.eval()
    return model

def get_hparams(yaml_path):
    with open(yaml_path, 'r') as f:
        hparams = Namespace(**yaml.safe_load(f))
    return hparams
    
if __name__ == '__main__':
    
    concat_flat_logits = lambda logits : torch.cat(list(map(lambda tensor : tensor.reshape(-1, tensor.shape[-1]), logits)), dim = 0)
    yaml_path = './experiments/moving_target/lstm/pretrained/default/version_2iyg9qr0/hparams.yaml'
    try:
        single_hparams = get_hparams(yaml_path)
        single_hparams.eval_batch_size = 1
        single_model = prep_fresh_load_model(single_hparams)
        
        large_hparams = get_hparams(yaml_path)
        large_hparams.eval_batch_size = 4
        large_model = prep_fresh_load_model(large_hparams)
        
        langs = ['Dutch']
        single_predictions = explore_predict.get_all_predictions(single_model, langs)
        large_predictions = explore_predict.get_all_predictions(large_model, langs)
        
        single_labels = explore_predict.get_padded_labels(single_model, 'Dutch')
        large_labels = explore_predict.get_padded_labels(large_model, 'Dutch')
        
        clean_labels = lambda labels : labels[labels != LABEL_PAD_ID]
        if not torch.all(clean_labels(single_labels) == clean_labels(large_labels)):
            import pdb; pdb.set_trace()
        
        single_logits = concat_flat_logits(single_predictions['Dutch']) 
        large_logits = concat_flat_logits(large_predictions['Dutch'])
        
        batch_types = ['single', 'large']
        clean_logits = {}
        for model, logits, labels, name in zip(
            [single_model, large_model],
            [single_logits, large_logits],
            [single_labels, large_labels],
            batch_types
        ):
            clean_logits[name] = model.set_padded_to_zero(labels, logits)
        
        for batch_type in batch_types:
            print(batch_type, clean_logits[batch_type].sum())
            
        import pdb; pdb.set_trace()
       
    except: import pdb; pdb.set_trace()