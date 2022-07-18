
import plotly.express as px
import glob

import yaml
import os
import json
import pandas as pd
import numpy as np

def get_heatmap(df, row_key, column_key, value_key, analysis_path):
    
    to_sorted_tick = lambda params : list(map(lambda s: str(s), sorted(list(set(params)))))
    df = df.sort_values(by=[row_key, column_key], ascending = True)
    row_values = to_sorted_tick(df[row_key])
    column_values = to_sorted_tick(df[column_key])
    result_values = df[value_key]
    
    number_of_rows = len(row_values)
    number_of_columns = len(column_values)
    
    raw_scores = np.array(result_values).reshape((number_of_rows, number_of_columns))
    rounded_scores = np.round_(raw_scores, decimals = 1)
    
    for modifier, scores in zip(['', 'rounded_'], [raw_scores, rounded_scores]):
        fig = px.imshow(
            scores, text_auto = True,
            labels = {'y' : row_key, 'x' : column_key, 'color' : value_key},
            y = row_values, x = column_values,
            color_continuous_scale = 'rainbow'
        )
        fig.write_html(os.path.join(analysis_path, f'{modifier}{value_key}.html'))
    return raw_scores
    
if __name__ == '__main__':
    
    analysis_path = './experiments/sweeps/latent_weights'
    weights = [1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 0][::-1]
    
    # Compose the dataframe first
    sweep_id = '3bfyjg41'
    extension = 'yaml'
    clean_extension = lambda s : s.split('/')[-1].replace(f'.{extension}', '')
    
    records = []
    config_paths = list(glob.glob(f'./wandb/sweep-{sweep_id}/*'))
    for config_path in config_paths:
        run_id = clean_extension(config_path).split('-')[-1]
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            kl, mse = config['latent_kl_weight']['value'], config['mse_weight']['value']
        run_paths = list(filter(lambda s: run_id in s, glob.glob(f'./wandb/*')))
        assert len(run_paths) == 1, f'{run_id}, {run_paths}'
        with open(os.path.join(run_paths[0], 'files', 'wandb-summary.json'), 'r') as f:
            results = json.load(f)
            for_df_results = { 'run_id' : run_id, 'latent_kl_weight' : kl, 'mse_weight' : mse }
            for phase in ['train', 'val']:
                for lang in ['English', 'Dutch']:
                    key = f'best_{phase}_{lang}_acc_epoch'
                    for_df_results[key] = results[key]
        records.append(for_df_results)
    df = pd.DataFrame.from_records(records)
    for phase in ['train', 'val']:
        for lang in ['English', 'Dutch']:
            key = f'best_{phase}_{lang}_acc_epoch'
            scores = get_heatmap(df, 'latent_kl_weight', 'mse_weight', key, analysis_path)
                