
import os
import subprocess
import glob

def make_single_script(to_run_scripts, index, script_folder):
    header = ['#!/bin/bash\n']
    commands = header + list(map(lambda s : s + '\n', to_run_scripts))
    script_folder = './src/scripts/baseline/parallel'
    script_path = os.path.join(script_folder, f'{index}.sh')
    if not os.path.exists(script_folder): os.makedirs(script_folder)
    with open(script_path, 'w') as f:
        f.writelines(commands)
    subprocess.run(['chmod', 'u+x', script_path])
    
if __name__ == '__main__':
    script_folder = './src/scripts/baseline'
    all_scripts = sorted(glob.glob(script_folder + '/*/*/*/*/*.sh'))
    
    size = len(all_scripts) // 3
    split_1, split_2, split_3 = all_scripts[:size], all_scripts[size:2*size], all_scripts[2*size:]
    for index, split in enumerate([split_1, split_2, split_3]):
        make_single_script(split, index, script_folder)
    