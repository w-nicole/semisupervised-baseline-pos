#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/latent_space/dutch"}
train_languages="Dutch"
val_languages="Dutch"
data_path=${2:-"../../ud-treebanks-v1.4"}

bs=16
ep=20

python3 src/train_latent_space.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --group "latent_space" \
    --job_type "base" \
    --name "dutch" \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --gpu 1 