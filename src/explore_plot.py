
# run from repository root

import matplotlib.pyplot as plt
import os
import torch
import numpy as np

from enumeration import Split
import constant
import util
from model import EncoderDecoder


def compare_priors(model, analysis_folder):
    
    english_smoothed_prior = model.get_smoothed_english_prior()
    raw_dutch_counts = model.get_label_counts("Dutch", Split.dev)
    
    dutch_prior = raw_dutch_counts / torch.sum(raw_dutch_counts)
    
    plt.title('Normalized label counts')
    labels = np.arange(len(constant.UD_POS_LABELS))
    plt.bar(labels, util.remove_from_gpu(dutch_prior).numpy().flat, color = 'r', alpha = 0.5, label = 'dutch')
    plt.bar(labels, util.remove_from_gpu(english_smoothed_prior).numpy().flat, color = 'g', alpha = 0.5, label = 'smoothed english')
    plt.savefig(os.path.join(analysis_folder, 'english_vs_dutch_counts.png'))
    
if __name__ == '__main__':
    model_folder = './experiments/decoder_baseline/dutch_trained_english_smoothed'
    checkpoint_name = 'ckpts_epoch=0-val_acc=76.034.ckpt'
    model_type = EncoderDecoder
    
    model = model_type.load_from_checkpoint(os.path.join(model_folder, 'ckpts', checkpoint_name))
    
    data_analysis_folder = './experiments/data_analysis'
    if not os.path.exists(data_analysis_folder):
        os.makedirs(data_analysis_folder)
    compare_priors(model, data_analysis_folder)
    
    
    