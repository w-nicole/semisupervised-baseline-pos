#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/token_effect/nll_token"}
train_languages="English Dutch"
val_languages="English Dutch"
data_path=${2:-"../ud-treebanks-v1.4"}

bs=16
ep=3

python3 src/train_with_token_loss.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --latent_size 64 \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --group "token_effect" \
    --job_type "explore" \
    --name "nll_token" \
    --mse_weight 0 \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --gpu 1 