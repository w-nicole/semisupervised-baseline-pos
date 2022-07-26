
import glob
import os
import json
import pandas as pd
import numpy as np
import scipy.stats

get_seed_keys = lambda seeds : list(map(lambda seed : f'seed={seed}', seeds))

def save_seed_sweep(seed_folder, seeds, lang):
    
    metric_key = f'best_val_{lang}_pos_acc_epoch'
    results = {}
    for seed in seeds:
        template = os.path.join(seed_folder, f'seed={seed}', 'wandb/run-*/files/wandb-summary.json')
        matches = glob.glob(template)
        assert len(matches) == 1, f'{template} {matches}'
        with open(matches[0], 'r') as f:
            results[f'seed={seed}'] = json.load(f)[metric_key]
    df = pd.DataFrame.from_records([results])
    
    seed_keys = get_seed_keys(seeds)
    all_runs = np.array(df[seed_keys])
    df['mean'] = all_runs.mean()
    df['stdev'] = all_runs.std()
    path = os.path.join(seed_folder, f'stats_{metric_key}.csv')
    df.to_csv(path)
    print(f'Analysis written to {path}')
    return df
    
def compare_seed_sweeps(base_folder, base_name_1, base_name_2, seeds, lang):
    results = {}
    for name in [base_name_1, base_name_2]:
        results[name] = save_seed_sweep(os.path.join(base_folder, name), seeds, lang)
    
    seed_keys = get_seed_keys(seeds)
    extract_seeds = lambda df, seed_keys : np.array(df[seed_keys]).squeeze()
    
    _, p_value = scipy.stats.ttest_rel(
        extract_seeds(results[base_name_1], seed_keys),
        extract_seeds(results[base_name_2], seed_keys)
    )

    report = {}
    report[f'{base_name_1}_to_{base_name_2}'] = p_value.item()
    
    with open(os.path.join(base_folder, f'{lang}_t_test.json'), 'w') as f:
        json.dump(report, f)
    
    return report
    
    


