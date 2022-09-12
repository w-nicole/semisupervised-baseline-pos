
import matplotlib.pyplot as plt
import os

import util
from enumeration import Split
from model import Tagger
import constant

if __name__ == '__main__':
    
    for subset_count in [5, 10, 50, 100, 500, 1000, 1500]:
        
        lang = 'English'
        
        model = util.get_subset_model(Tagger, False, subset_count)
        train_counts = model.get_label_counts(lang, Split.train).cpu()
        val_counts = model.get_label_counts(lang, Split.dev).cpu()
        
        plt.figure()
        
        range_labels = range(len(constant.UD_POS_LABELS))
        plt.title(f'Label distribution, {lang}')
        plt.bar(range_labels, train_counts.numpy() / train_counts.sum(), color = 'g', alpha = 0.5, label = 'train')
        plt.bar(range_labels, val_counts.numpy() / val_counts.sum(), color = 'r', alpha = 0.5, label = 'val')
        plt.legend()
        plt.xlabel('Label'); plt.ylabel('Normalized frequency')
        plt.xticks(range_labels, constant.UD_POS_LABELS, rotation = 45)
        
        base_path = '../../alt/semisupervised-baseline-pos/'
        sweep_path = os.path.join(base_path, 'experiments/subset/dataset_analysis')
        analysis_path = os.path.join(sweep_path, f'subset_count={subset_count}')
        if not os.path.exists(analysis_path): os.makedirs(analysis_path)
        figure_path = os.path.join(analysis_path, f'label_distribution_{lang}.png')
        plt.savefig(figure_path)
        
        