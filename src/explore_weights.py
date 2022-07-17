
import os
import torch

import yaml
from argparse import Namespace
import torch
import torch.nn.functional as F
import numpy as np

from model import LatentToPOS

import pytorch_lightning as pl

if __name__ == '__main__':
    
    hparams_path = './experiments/moving_target/lstm/pretrained/nll_kl_0.001/default/version_3rp6tkzs'
    with open(os.path.join(hparams_path, 'hparams.yaml'), 'r') as f:
        hparams = Namespace(**yaml.safe_load(f))
    raw_model = LatentToPOS(hparams)
    
    ckpt_path = './experiments/moving_target/lstm/pretrained/nll_kl_0.001/default/version_3rp6tkzs/ckpts/ckpts_epoch=0-val_Dutch_acc_epoch=21.837.ckpt'
    trained_model = LatentToPOS.load_from_checkpoint(ckpt_path)

    for model_name, model in zip(['raw', 'trained'], [raw_model, trained_model]):
        print(f'Next model: {model_name}')
        try:
            layers = [
                ('encoder/mean', model.encoder_mu),
                ('encoder/sigma', model.encoder_log_sigma),
                ('pos', model.decoder_pos.linear),
                ('reconstruction', model.decoder_reconstruction.linear)
            ]
                
            l2 = lambda tensor : tensor.pow(2).sum().sqrt()
            for layer_name, layer in layers:
                print(f'\tLayer {layer_name}')
                print(f'\t\tWeight: {l2(layer.weight)}')
                print(f'\t\tBias: {l2(layer.bias)}')
        except: import pdb; pdb.set_trace()
