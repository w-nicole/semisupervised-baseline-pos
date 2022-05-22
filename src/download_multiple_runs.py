
import os
import glob
import shutil

if __name__ == '__main__':
    
    base_path = os.path.join('../experiments', 'udpos', '0-shot-finetune-freeze-1', 'bert-base-multilingual-cased')
    all_results_folders = glob.glob(f'{base_path}/*/*')
    
    for folder in all_results_folders:
        new_folder = folder.replace('udpos', 'udpos_results_only_multiple_runs')
        filename = 'results.jsonl'
        os.makedirs(new_folder)
        shutil.copy(os.path.join(folder, filename), os.path.join(new_folder, filename))
    