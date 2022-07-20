
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
    
def find_hparam(hparam_name, hparam_list):
    hparam_prefix = f'--{hparam_name}='
    matches = list(filter(lambda s : hparam_prefix in s, hparam_list))
    assert len(matches) == 1, f'{hparam_name}, {hparam_list}, {matches}'
    return matches[0].replace(hparam_prefix, "")
    
    
if __name__ == '__main__':
    

    sweep_id, sweep_name = '', ''
    analysis_path = f'./experiments/sweeps/{sweep_name}'

    extension = 'yaml'
    clean_extension = lambda s : s.split('/')[-1].replace(f'.{extension}', '')
    
    records = []
    config_paths = list(glob.glob(f'./experiments/sweeps/{sweep_name}/*/wandb/run-*/files'))
    for config_folder in config_paths:
        with open(os.path.join(config_folder, 'wandb-metadata.json'), 'r') as f:
            config = json.load(f)["args"]
            kl, mse = float(find_hparam('latent_kl_weight', config)), float(find_hparam('mse_weight', config))
        with open(os.path.join(config_folder, 'wandb-summary.json'), 'r') as f:
            results = json.load(f)
            for_df_results = {'latent_kl_weight' : kl, 'mse_weight' : mse }
            for phase in ['train', 'val']:
                for lang in ['English', 'Dutch']:
                    key = f'best_{phase}_{lang}_acc_epoch'
                    for_df_results[key] = results[key]
        records.append(for_df_results)
    import pdb; pdb.set_trace()
    df = pd.DataFrame.from_records(records)
    for phase in ['train', 'val']:
        for lang in ['English', 'Dutch']:
            key = f'best_{phase}_{lang}_acc_epoch'
            scores = get_heatmap(df, 'latent_kl_weight', 'mse_weight', key, analysis_path)
                