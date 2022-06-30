#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory where this is run
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, variables, options, etc.
# Added hyperparameters that yield best performance on English val for last layer finetuned BERT.
# Edited arguments.
# Edited to run the train_decoder_base script instead.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${4:-"./experiments/debug/auxiliary"}

src="English"
tgt="English"

data_path=${5:-"../ud-debug"}
encoder_checkpoint=${6:-"./experiments/normal/phase_1/version_0/ckpts/ckpts_epoch=2-val_English_acc=96.807.ckpt"}

bs=64
ep=1000
lr=1e-1

python3 src/train_decoder_base.py \
    --seed "$seed" \
    --task "$task" \
    --data_dir "$data_path" \
    --trn_langs $src \
    --val_langs $tgt \
    --pretrain "$model" \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_save_path "$save_path" \
    --exp_name memorize \
    --gpus 1 \
    --encoder_checkpoint "$encoder_checkpoint" \
    --auxiliary_hidden_layers 0 \
    --auxiliary_hidden_size 2 \
    --auxiliary_size 2 \
    --mse_weight 0