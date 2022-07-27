#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/debug/memorize_single_with_token"}
train_languages="Dutch"
val_languages="Dutch"
data_path=${2:-"../ud-debug-single"}

bs=16
ep=1000

python3 src/train_with_token_loss.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --latent_size 768 \
    --mse_weight 0 \
    --pos_nll_weight 0 \
    --debug_model_all_eval "y" \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --group "debug" \
    --job_type "memorize" \
    --name "with_token" \
    --patience $ep \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --gpu 1 