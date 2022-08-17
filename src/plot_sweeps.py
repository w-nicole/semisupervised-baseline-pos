
import glob

import yaml
import os
import json
import pandas as pd
import numpy as np
    
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
                for_df_results = {k : sweeps.find_hparam(k, config, cast)  for k, cast in zip(searched_hparams, cast_as) }
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
                scores = sweeps.get_heatmap(df, searched_hparams[0], searched_hparams[1], key, analysis_path)
            elif len(searched_hparams) == 1:
                scores = sweeps.get_heatmap(df, 'for_1d', searched_hparams[0], key, analysis_path)
            else:
                assert False, f"{len(searched_hparams)} number of hparams searched not supported by this plotting code."
                
    
if __name__ == '__main__':
    
    sweep_name = 'subset_number_mse_0'
    grid_hparams = ['subset_count']
    cast_as = [int]
    langs = ['English']
   
    sweep_path = f'./experiments/sweeps/{sweep_name}'
    modifier = ''
    template = os.path.join(sweep_path, f'{modifier}*/wandb/run-*/files')
    heatmap_matches(template, grid_hparams, os.path.join(sweep_path, 'heatmaps', modifier), cast_as, langs)
    
    