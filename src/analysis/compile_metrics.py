
import json
import os
import pandas as pd
import glob
import numpy as np

import datetime


def load_results(experiment_path):
    
    all_results = []
    with open(os.path.join(experiment_path, 'results.jsonl'), 'r') as f:
        for line in f.readlines():
            all_results.append(json.loads(line))
    
    return all_results


def load_evaluation_results(experiment_path, phase):
    
    assert phase in {'val', 'tst'}, f"Invalid phase name: {phase}, specify either 'val' or 'test'."
    results = load_results(experiment_path)
    which_index = -2 if phase == 'val' else -1
    if phase == 'tst':
        return results[-1]
    # Find the max validation English accuracy that occured and return that as the representative,
    # which is used in grid searching.
    best_entry = sorted(results[:-1], key = lambda entry : entry['val_English_acc'])[-1]
    return best_entry


def generate_accuracy_dict(experiment_path, phase, language_list):
    
    assert phase in {'tst', 'val'}
    
    results = load_evaluation_results(experiment_path, phase)
    keys = list(map(lambda language : f'{phase}_{language}_acc', language_list))
    
    extract_language = lambda key : key.split('_')[1]
    return { extract_language(key) : results[key] for key in keys }

def generate_accuracy_table(experiment_path, phase):
    
    return pd.DataFrame.from_records([generate_accuracy_dict(experiment_path, phase)])

def compare_results_to_paper(results):
    
    # Numbers should match the last row of Table 5, p. 7
    # https://arxiv.org/pdf/1904.09077.pdf
    expected_results = [87.4, 88.3, 89.8, 97.1, 85.2, 72.8, 83.2, 84.7, 75.9, 86.9, 82.1, 84.7, 83.6, 84.2, 91.3]
    
    results_difference = {}
    for key, expected in zip(results, expected_results):
        results_difference[key] = results[key] - expected
        
    return results_difference
        
    
def get_all_baseline_difference_results(base_path):
    
    all_results_paths = glob.glob(f'{base_path}/*/*')
    all_results = {}
    for result_path in all_results_paths:
        key = result_path.split('/')[-2] # The folder with the parameters
        results = generate_accuracy_dict(result_path, 'tst', False)
        # Prevent duplicates 
        if key in all_results: key += str(datetime.datetime.now())
        all_results[key] = compare_results_to_paper(results)
        
    return all_results    

def get_all_baseline_smallest_difference_results(base_path):
    
    all_differences = get_all_baseline_difference_results(base_path)
    
    sort_results = lambda key : np.mean(np.abs(np.array(list(all_differences[key].values()))))
    
    best_key = sorted(all_differences.keys(), key = sort_results, reverse = True)[-1]
    
    return best_key, all_differences[best_key]

