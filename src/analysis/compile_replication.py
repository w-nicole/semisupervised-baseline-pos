
# Messier code, remove/clean up before finalizing

from collections import defaultdict
import pandas as pd
import os
import numpy as np

from analysis import compile_metrics

import glob

def analyze_grid_run(name_path_section, seeds):

    all_languages = ["Bulgarian", "Danish", "German", "English", "Spanish", "Persian", "Hungarian", "Italian", "Dutch", "Polish", "Portuguese", "Romanian", "Slovak", "Slovenian", "Swedish"]
    
    version_metrics = defaultdict(int)
    best_version_keys = defaultdict(str)
    
    # Total expected: 12
    represented_metrics = defaultdict(set)

    for seed in seeds:

        seed_path = f'../experiments/{name_path_section}'
        hyperparameter_search_path = os.path.join(seed_path, '0-shot-finetune-freeze-1', 'bert-base-multilingual-cased')

        current_version_metrics = {}
        hyperparameter_version_base_path = f'{hyperparameter_search_path}/*/version_0'

        hyperparameter_paths = glob.glob(hyperparameter_version_base_path)

        for path in hyperparameter_paths:
            represented_metrics[seed] |= {path.split('/')[-2]}
            best_validation_entry = compile_metrics.load_evaluation_results(path, 'val')
            best_validation_accuracy = best_validation_entry['val_English_acc']
            current_version_metrics[path] = best_validation_accuracy

        all_keys = sorted(current_version_metrics.keys())

        # Look for the maximum value amongst the grid search values
        sorted_key_indices = np.argsort(list(map(lambda entry : current_version_metrics[entry], all_keys)))
        best_key_index = sorted_key_indices[-1]
        best_key = all_keys[best_key_index]
        best_version_keys[seed] = best_key

        test_accuracies = compile_metrics.generate_accuracy_dict(best_key, 'tst', all_languages)
        paper_difference = compile_metrics.compare_results_to_paper(test_accuracies)

        version_metrics[seed] = (test_accuracies, paper_difference)

        best_validation_accuracy = compile_metrics.load_evaluation_results(best_version_keys[42], 'val')

        return version_metrics, best_version_keys[42], best_validation_accuracy

def compare_grid_searches(reference, rerun):
    """Accepts the outputs of analyze_grid_run, one per set in the arguments"""
    
    reference_absolute, reference_diff = reference[0][42]
    rerun_absolute, rerun_diff = rerun[0][42]
    
    comparison = defaultdict(list)
    for lang in reference_absolute:
        comparison['ref_absolute'].append(reference_absolute[lang])
        comparison['run_absolute'].append(rerun_absolute[lang])
        comparison['ref_diff_to_paper'].append(reference_diff[lang])
        comparison['run_diff_to_paper'].append(rerun_diff[lang])
        comparison['run - ref'].append(rerun_absolute[lang] - reference_absolute[lang])
        comparison['lang'].append(lang)
                                               
    column_order = ['lang', 'ref_absolute', 'run_absolute', 'ref_diff_to_paper', 'run_diff_to_paper', 'run - ref']
    df = pd.DataFrame.from_records(comparison)[column_order]
    return df
    


