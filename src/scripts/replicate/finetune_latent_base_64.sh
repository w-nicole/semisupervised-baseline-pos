#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory where this is run
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, options, etc.
# Added hyperparameters that yield best performance on English val for last layer finetuned BERT.
# Edited arguments.
# Edited to run the train_encoder script instead.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)
save_path=${4:-"./experiments/replicate/finetune_64"}

train_languages="English"
val_languages="English Dutch"
data_path=${5:-"../ud-treebanks-v1.4"}

bs=16
ep=3

python3 src/train_latent_base.py \
    --seed "$seed" \
    --task "$task" \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --target_language "English" \
    --pretrain "$model" \
    --mse_weight 0 \
    --latent_size 64 \
    --batch_size $bs \
    --max_epochs $ep \
    --group "replicate" \
    --job_type "reference" \
    --name "finetune_64" \
    --warmup_portion 0.1 \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --gpus 1
    