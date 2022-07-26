#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory where this is run
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, options, etc.
# Added hyperparameters that yield best performance on English val for last layer finetuned BERT.
# Edited arguments.
# Edited to run the train_encoder script instead.

model=${1:-"bert-base-multilingual-cased"}
task=${2:-"udpos"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)
save_path=${3:-"./experiments/debug/t_test/dutch_mse_script"}

train_languages="English Dutch"
val_languages="English Dutch"
data_path=${4:-"../ud-treebanks-v1.4"}

bs=16
ep=3
seed=2

python3 src/train_latent_base.py \
    --seed "$seed" \
    --task "$task" \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --target_language "Dutch" \
    --pretrain "$model" \
    --mse_weight 0 \
    --latent_size 64 \
    --batch_size $bs \
    --max_epochs $ep \
    --group "dutch_mse_script_t_test" \
    --job_type "compare" \
    --name "seed=$seed" \
    --warmup_portion 0.1 \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --gpus 1
    