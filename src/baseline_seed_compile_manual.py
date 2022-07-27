    
import os
import glob
import json
import pandas as pd
import numpy as np

from compare import significance

def save_seed_sweeps(seed_folder, seeds, lang):
    metric_key = f'best_val_{lang}_pos_acc_epoch'
    results = {}
    for seed in seeds:
        template = os.path.join(seed_folder, f'seed={seed}', 'wandb-summary.json')
        matches = glob.glob(template)
        assert len(matches) == 1, f'{template} {matches}'
        with open(matches[0], 'r') as f:
            results[f'seed={seed}'] = json.load(f)[metric_key]
    df = pd.DataFrame.from_records([results])
    
    seed_keys = significance.get_seed_keys(seeds)
    all_runs = np.array(df[seed_keys])
    df['mean'] = all_runs.mean()
    df['stdev'] = all_runs.std()
    path = os.path.join(seed_folder, f'stats_{metric_key}.csv')
    df.to_csv(path)
    print(f'Analysis written to {path}')
    return df

if __name__ == '__main__':
    
    seed_folder = './experiments/sweeps/seed'
    seeds = [42, 0, 1, 2, 3]
    for lang in ['English', 'Dutch']:
        save_seed_sweeps(seed_folder, seeds, lang)
    
