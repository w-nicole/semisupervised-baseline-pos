
import pandas as pd
import os
import json
import numpy as np

if __name__ == '__main__':
    baseline_df = pd.read_csv('/nobackup/users/wongn/crosslingual-nlp-private/experiments/seed_variation/val_accuracy_across_seeds.csv')
    ours_folder = '/nobackup/users/wongn/alt/semisupervised-baseline-pos/experiments/multiple_langs/all_langs/all_langs/'
    summary_path = os.path.join(ours_folder, 'wandb/run-20220729_224837-1z2mncu7/files/', 'wandb-summary.json')
    
    languages = [
        'Bulgarian', 'Danish', 'German', 'English', 'Spanish', 'Persian',
        'Hungarian', 'Italian', 'Dutch', 'Polish', 'Portuguese', 'Romanian',
        'Slovak', 'Slovenian', 'Swedish'
    ]
    baseline_mean = baseline_df[languages][baseline_df.type == 'mean'] * 100
    baseline_stdev = baseline_df[languages][baseline_df.type == 'stdev'] * 100
    
    with open(summary_path, 'r') as f:
        raw_results = json.load(f)
        our_results = { lang : raw_results[f'best_val_{lang}_pos_acc_epoch'] for lang in languages }
        our_df = pd.DataFrame.from_records([our_results])
        
    our_df['type'] = 'ours-seed=42'
    baseline_mean['type'] ='theirs-mean'
    baseline_stdev['type'] ='theirs-stdev'
    difference = baseline_mean[languages].reset_index() - our_df[languages].reset_index()
    difference['type'] = 'theirs-our_df'
    
    report = pd.concat([our_df, baseline_mean, baseline_stdev, difference])
    report.to_csv(os.path.join(ours_folder, 'val_acc_against_baseline_stats.csv'))
    
    
    