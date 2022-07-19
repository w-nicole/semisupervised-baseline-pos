#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/replicate"}
train_languages="English"
val_languages="English Dutch"
data_path=${2:-"../ud-treebanks-v1.4"}

bs=16
ep=3
lr=5e-5
pos_hidden_layers=-1
pos_hidden_size=0
mbert_hidden_size=-1
mbert_hidden_layers=0

python3 src/train_changing_target.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --latent_size 768 \
    --batch_size $bs \
    --latent_kl_weight 0 \
    --mse_weight 0 \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --target_language "English" \
    --default_save_path "$save_path" \
    --wandb_group "replicate" \
    --reconstruction_model_type "fcn" \
    --pos_model_type "fcn" \
    --gpus 1 \
    --freeze_mbert "n" \
    --debug_without_sampling "n" \
    --english_alone_as_supervised "y"