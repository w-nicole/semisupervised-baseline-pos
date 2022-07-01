
# Parts of the sections written to bash script adapted from Shijie Wu's crosslingual-nlp repository.
# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changin, removing)
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

# See LICENSE in this codebase for license information.

import json
import subprocess
import os

def create_single_script(language, mbert_type, encoder_type, decoder_type, kl, exp_name, group, job_type):
    content_path = './script_content'
    with open(os.path.join(content_path, 'paths.json'), 'r') as f:
        paths = json.load(f)
    with open(os.path.join(content_path, 'header.txt'), 'r') as f:
        header = f.readlines()
    with open(os.path.join(content_path, 'hyperparams.json'), 'r') as f:
        hyperparams = json.load(f)
        
    exp_path = f'baseline/{language}/mbert_{mbert_type}/encoder_{encoder_type}/decoder_{decoder_type}/kl_{kl}'
    save_path = os.path.join('./experiments', exp_path)
    mbert_key = f'mbert_{mbert_type}'
    
    if mbert_key == 'mbert_fixed':
        concat_string = "y"
    elif mbert_key == 'mbert_pretrained':
        concat_string = "n"
    else:
        assert False, f"Invalid mbert type: {mbert_type}"
    
    checkpoints = {}
    checkpoints['mbert'] = '' if mbert_type == 'fixed' else paths['encoder']['mbert_pretrained']
    checkpoints['encoder'] = '' if encoder_type == 'random' else paths['encoder'][mbert_key]
    checkpoints['decoder'] = '' if decoder_type == 'random' else paths['decoder'][mbert_key]
    
    format_to_braces = lambda name, index, value : f'{name}=${{{index}:-"{value}"}}'
    pre_command_variables = [
        format_to_braces('save_path', 1, save_path),
        f'train_languages="{language}"',
        f'val_languages="English Dutch"',
        format_to_braces('data_path', 2, paths['data_path']),
        format_to_braces('mbert_checkpoint', 3, checkpoints['mbert']),
        format_to_braces('encoder_checkpoint', 4, checkpoints['encoder']),
        format_to_braces('decoder_checkpoint', 5, checkpoints['decoder']),
        f'bs={hyperparams["batch_size"]}',
        f'ep={hyperparams["epochs"]}',
        f'lr={hyperparams["learning_rate"]}',
        f'\npython3 src/train_decoder.py \\'
    ]
    command_variables = [
        f'\t--wandb_group {group}',
        f'\t--wandb_job_type {job_type}'
        f'\t--data_dir "$data_path"',
        f'\t--trn_langs $train_languages',
        f'\t--val_langs $val_languages',
        f'\t--batch_size $bs',
        f'\t--learning_rate $lr',
        f'\t--max_epochs $ep',
        f'\t--warmup_portion 0.1',
        f'\t--default_save_path "$save_path"',
        f'\t--exp_name {exp_name}',
        f'\t--gpus 1',
        f'\t--freeze_mbert "y"',
        f'\t--concat_all_hidden_states "{concat_string}"'
    ]
    for checkpoint_type, path in checkpoints.items():
        if path:
            command_variables.append(f'\t--{checkpoint_type}_checkpoint ${checkpoint_type}_checkpoint')
    command_variables.extend([
        f'\t--pos_kl_weight {kl}',
        f'\t--schedule "reduceOnPlateau"',
        f'\t--prior_type "optimized_data"'
    ])
    command_variables = list(map(lambda s : s + " \\", command_variables[:-1])) + [command_variables[-1]]
    body = list(map(lambda s : s + '\n', pre_command_variables + command_variables))
    text = header + ['\n'] + body
    
    
    script_path = os.path.join('./src/scripts', exp_path + '.sh')
    script_folder = '/'.join(script_path.split('/')[:-1])
    if not os.path.exists(script_folder):
        os.makedirs(script_folder)
    with open(script_path, 'w') as f:
        f.writelines(text)
    subprocess.run(['chmod', 'u+x', script_path])
    
if __name__ == '__main__':
        
    # Add the necessary languages
    exp_name = 'linear'
    group = 'pure_baseline'
    for train_language in ['English', 'Dutch']:
        for mbert in ['fixed', 'pretrained']:
            for encoder in ['pretrained']:
                for decoder in ['random', 'pretrained']:
                    for kl in [0, 0.1, 1]:
                        print(f'Processing: {train_language}, {mbert}, {encoder}, {decoder}, {kl}')
                        create_single_script(train_language, mbert, encoder, decoder, kl, exp_name, group, train_language)
    