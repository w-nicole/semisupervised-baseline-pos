
import glob
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import pandas as pd

from model import Single
from enumeration import Split
from predict import across, predict_utils

if __name__ == '__main__':
    
    # Doesn't run in a single run, but only need to generate once.
    
    languages = ['English', 'Dutch']
    epochs = 40
    subset_count = 10
    all_epochs = list(range(epochs))
    
    mask_metrics = defaultdict(dict)
    
    base_folder = '../../alt/semisupervised-baseline-pos/experiments'
    sweep_folder = os.path.join(base_folder, 'subsets_over_epochs')
    
    # for is_masked in [True, False]:
    #     hparam_folder = os.path.join(sweep_folder, f'masked={is_masked}')
    #     for phase in [Split.train]:
    #         dataloaders_dict, padded_labels_dict = predict_utils.get_over_languages_args(
    #                 Single, phase, languages, subset_count = subset_count
    #             )        
    #         for epoch in all_epochs:
    #             current_template = os.path.join(hparam_folder, f'version_*/ckpts/ckpts_epoch={epoch}-*.ckpt')
    #             matches = glob.glob(current_template)
    #             if len(matches) == 0:
    #                 print(f'Skipping: {current_template}')
    #                 continue
    #             if len(matches) > 1:
    #                 import pdb; pdb.set_trace()
    #             if not len(matches) == 1: import pdb; pdb.set_trace()
    #             checkpoint_path = matches[0]
    #             analysis_path = predict_utils.get_analysis_path(checkpoint_path)
    #             df = across.predict_over_languages(
    #                     checkpoint_path, Single, phase, languages,
    #                     dataloaders_dict[is_masked], padded_labels_dict[is_masked], 'subset'
    #                 )
    #             mask_metrics[is_masked][epoch] = { f'{phase}_{lang}' : df[lang] for lang in languages }
                
    # Compilation only, assume the above runs
    for is_masked in [True, False]:
        hparam_folder = os.path.join(sweep_folder, f'masked={is_masked}')
        mask_metrics[is_masked] = defaultdict(dict)
        for phase in [Split.train, Split.dev]:
            for epoch in all_epochs:
                current_template = os.path.join(hparam_folder, f'version_*/ckpts/ckpts_epoch={epoch}-*.ckpt')
                matches = glob.glob(current_template)
                if len(matches) == 0:
                    print(f'Skipping: {current_template}')
                    continue
                if len(matches) > 1:
                    import pdb; pdb.set_trace()
                if not len(matches) == 1: import pdb; pdb.set_trace()
                checkpoint_path = matches[0]
                analysis_path = predict_utils.get_analysis_path(checkpoint_path)
                modifier = 'real' if phase == Split.dev else 'subset'
                df = pd.read_csv(os.path.join(analysis_path, f'{modifier}_{phase}_predictions', f'{modifier}_{phase}_accuracies.csv'))
                for lang in languages:
                    mask_metrics[is_masked][epoch][f'{phase}_{lang}'] = df[lang].item()
  
    plot_folder = os.path.join(sweep_folder, 'plots')
    for phase in [Split.train, Split.dev]:
        for lang in languages:
            
            modifier = f'{phase}_{lang}'
            figure_path = os.path.join(plot_folder, f'{modifier}.png')
            
            plt.title(modifier)
            masked_accuracies = [ mask_metrics[True][epoch][modifier] for epoch in range(len(mask_metrics[True])) ]
            unmasked_accuracies = [ mask_metrics[False][epoch][modifier] for epoch in range(len(mask_metrics[False])) ]
            plt.plot(range(len(masked_accuracies)), masked_accuracies, color = 'r', label = 'masked')
            plt.plot(range(len(unmasked_accuracies)), unmasked_accuracies, color = 'g', label = 'unmasked')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend()
            if not os.path.exists(plot_folder): os.makedirs(plot_folder)
            plt.savefig(figure_path)
            plt.figure()
                
    
            
                