
import os
import torch

import yaml
from argparse import Namespace
from model import VAE
import torch
import torch.nn.functional as F
import numpy as np

from metric import NMIMetric

if __name__ == '__main__':
    with open('./scratchwork/dutch_only_hparams.yaml', 'r') as f:
        hparams = Namespace(**yaml.safe_load(f))
    hparams.prior_type = 'optimized_data'
    model = VAE(hparams).cpu()
    # dutch_first_batch = next(model.train_dataloader().__iter__())
    # dutch_first_batch = {k : v.cuda() if isinstance(v, torch.Tensor) else v for k, v in dutch_first_batch.items()}
    
    # outputs = model(dutch_first_batch)

    classes = 18
    metric = NMIMetric(classes)
    # Example: 0 MI
    size = (300,)
    
    get_labels = lambda size, classes : torch.from_numpy(np.random.randint(0, classes, size = size))
    padded_labels = get_labels(size, classes)
    encoder_log_probs = F.one_hot(get_labels(size, classes), num_classes = classes).float()
    metric.add(padded_labels, encoder_log_probs)
    low_nmi = metric.get_metric()
    
    #Example: max MI (that is, min(H(X), H(Y))
    metric = NMIMetric(classes)
    padded_labels = get_labels(size, classes)
    encoder_log_probs = F.one_hot((padded_labels + 1) % classes, num_classes = classes).float()
    metric.add(padded_labels, encoder_log_probs)
    high_nmi = metric.get_metric()
    
    
    
    