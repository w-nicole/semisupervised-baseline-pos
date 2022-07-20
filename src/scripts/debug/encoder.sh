#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/debug/encoder"}
train_languages="Dutch"
val_languages="Dutch"
data_path=${2:-"../../ud-debug-single"}

bs=16
ep=3
lr=5e-3
latent_size=64

python3 src/train_fixed_target.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --subset_ratio 0.1 \
    --learning_rate $lr \
    --number_of_workers 1 \
    --subset_ratio 1 \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --wandb_group "debug" \
    --default_save_path "$save_path" \
    --encoder_mu_model_type "linear" \
    --encoder_log_var_model_type "mlp" \
    --gpus 1 \
    --freeze_mbert "y"