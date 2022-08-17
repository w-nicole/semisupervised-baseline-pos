
import glob
import pandas as pd
import os

if __name__ == '__main__':
    sweep_folder = '../../alt/semisupervised-baseline-pos/experiments/subset/masked/'
    template = 'subset_count=*/version_*/ckpts_epoch=*/val_predictions/val_accuracies.csv'
    
    paths = glob.glob(os.path.join(sweep_folder, template))
    subset_counts = list(map(lambda path : int(path.split('/')[7].split('=')[-1]), paths))

    all_df = pd.concat([pd.read_csv(path) for path in paths]) * 100.0
    all_df['subset_count'] = subset_counts
    all_df = all_df.sort_values(by='subset_count')
    
    all_df.to_csv(os.path.join(sweep_folder, 'val_accuracies_across_langs.csv'))
    all_df.round(decimals=3).to_csv(os.path.join(sweep_folder, 'rounded_val_accuracies_across_langs.csv'))
    