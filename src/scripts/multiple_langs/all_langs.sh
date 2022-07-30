#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/multiple_langs/all_langs"}
train_languages="Bulgarian Danish German English Spanish Persian Hungarian Italian Dutch Polish Portuguese Romanian Slovak Slovenian Swedish"
val_languages="Bulgarian Danish German English Spanish Persian Hungarian Italian Dutch Polish Portuguese Romanian Slovak Slovenian Swedish"
data_path=${2:-"../../ud-treebanks-v1.4"}

bs=16
ep=3
latent_size=64

python3 src/train_latent_base.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --number_of_workers 1 \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --group "multiple_langs" \
    --job_type "sweep" \
    --name "all_langs" \
    --mse_weight 1e-6 \
    --default_save_path "$save_path" \
    --gpus 1 \
    --freeze_mbert "n"