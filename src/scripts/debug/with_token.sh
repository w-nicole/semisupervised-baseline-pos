#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/debug/run_with_token"}
train_languages="English"
val_languages="English"
data_path=${2:-"../ud-treebanks-v1.4"}

bs=16
ep=1
latent_size=2

python3 src/train_with_token_loss.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --subset_ratio 0.00001 \
    --number_of_workers 1 \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_learning_rate 5e-3 \
    --mbert_learning_rate 5e-5 \
    --group "debug" \
    --job_type "run" \
    --name "with_token" \
    --default_save_path "$save_path" \
    --freeze_mbert "y"