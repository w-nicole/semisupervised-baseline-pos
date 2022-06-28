
import os
import pandas as pd
import json
from collections import defaultdict

if __name__ == '__main__':
    
    paths = {
        'normal' : 'normal/phase_1/version_0',
        'frozen_concat' : 'frozen_concat/phase_2/lr_1e-2_bs_16/version_0',
        'large_frozen_encoder' : 'large_frozen_encoder/phase_1/lr_1e-3_size_768_layers_/version_0'
    }
    
    assert len(set(paths.keys())) == len(paths)
    
    analysis_path = './experiments/data_analysis/centroid_baseline'
    compilation = defaultdict(list)   
    
    for model_type, path_part in paths.items():
        stats_folder = os.path.join(analysis_path, path_part)
        model_record = compilation[model_type]
        model_record.append(model_type)
        for language in ['English', 'Dutch']:
            for phase in ['train', 'val']:
                modifier = f'{phase}_{language}'
                with open(os.path.join(stats_folder, f"{modifier}.json"), 'r') as f:
                    metric = json.load(f)['all']
                with open(os.path.join(stats_folder, f"standard_{modifier}.json"), 'r') as f:
                    standard_metric = json.load(f)['all']
                model_record[modifier].append(metric)
                model_record[f'standard_{modifier}'].append(standard_metric)
    
    pd.DataFrame.to_csv(os.path.join(analysis_path, 'across_models.csv'))
                
                
    
    
    
    