#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory where this is run
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, variables, options, etc.
# Added hyperparameters that yield best performance on English val for last layer finetuned BERT.
# Edited arguments.
# Edited to run the train_decoder script instead.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}
freeze=${4:-"12"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${5:-"./experiments"}

langs=(English Dutch)
data_path=${6:-"../ud-treebanks-v1.4"}
encoder_checkpoint=${7:-"./experiments/encoder_finetune_last_layer/version_0"}

bs=16
ep=3
lr=5e-5

python3 src/train_decoder.py \
    --seed "$seed" \
    --task "$task" \
    --data_dir "$data_path" \
    --trn_langs "${langs[@]}" \
    --val_langs "${langs[@]}" \
    --pretrain "$model" \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_save_path "$save_path" \
    --exp_name test_debug \
    --gpus 1 \
    --encoder_checkpoint "$encoder_checkpoint"
    
