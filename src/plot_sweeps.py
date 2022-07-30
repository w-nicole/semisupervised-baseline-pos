
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
        if not os.path.exists(analysis_path): os.makedirs(analysis_path)
        fig.write_html(os.path.join(analysis_path, f'{modifier}{value_key}.html'))
    return raw_scores
    
def find_hparam(hparam_name, hparam_list, cast_as):
    hparam_prefix = f'--{hparam_name}='
    matches = list(filter(lambda s : hparam_prefix in s, hparam_list))
    assert len(matches) == 1, f'{hparam_name}, {hparam_list}, {matches}'
    return cast_as(matches[0].replace(hparam_prefix, ""))
    
def heatmap_matches(template, searched_hparams, analysis_path, cast_as, langs):

    records = []
    
    config_paths = list(glob.glob(template))
    get_metric_key = lambda phase, lang : f'best_{phase}_{lang}_pos_acc_epoch'
    
    for config_folder in config_paths:
        with open(os.path.join(config_folder, 'wandb-metadata.json'), 'r') as metadata:
            config = json.load(metadata)["args"]
            summary_path = os.path.join(config_folder, 'wandb-summary.json')
            with open(summary_path, 'r') as summary:
                results = json.load(summary)
                for_df_results = {k : find_hparam(k, config, cast)  for k, cast in zip(searched_hparams, cast_as) }
                for_df_results['for_1d'] = 'default'
                for phase in ['train', 'val']:
                    for lang in langs:
                        key = get_metric_key(phase, lang)
                        for_df_results[key] = results[key]
                records.append(for_df_results)

    df = pd.DataFrame.from_records(records)

    for phase in ['train', 'val']:
        for lang in langs:
            key = get_metric_key(phase, lang)
            if len(searched_hparams) == 2:
                scores = get_heatmap(df, searched_hparams[0], searched_hparams[1], key, analysis_path)
            elif len(searched_hparams) == 1:
                scores = get_heatmap(df, 'for_1d', searched_hparams[0], key, analysis_path)
            else:
                assert False, f"{len(searched_hparams)} number of hparams searched not supported by this plotting code."
                
    
if __name__ == '__main__':
    
    sweep_id, sweep_name = 'ayg4hf9c', 'turkish/decoder'
    grid_hparams = ['pos_model_type', 'reconstruction_model_type']
    cast_as = [str, str]
    langs = ['English', 'Turkish']
   
    sweep_path = f'./experiments/sweeps/{sweep_name}'
    
    for weight in [1e-6, 1e-4, 1e-2, 1.0]:
        modifier = ''
        template = os.path.join(sweep_path, f'{modifier}*/wandb/run-*/files')
        heatmap_matches(template, grid_hparams, os.path.join(sweep_path, 'heatmaps', modifier), cast_as, langs)
    
    