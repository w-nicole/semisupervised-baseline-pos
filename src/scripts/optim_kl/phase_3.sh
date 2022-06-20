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

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${4:-"./experiments/debug_optim_kl"}

src="English Dutch"
tgt="English Dutch"
data_path=${5:-"../ud-treebanks-v1.4"}
decoder_checkpoint=${6:-"./experiments/debug_optim_kl/phase_2/version_0/ckpts/ckpts_epoch=3-val_English_decoder_loss=85.536.ckpt"}

bs=16
ep=15
lr=1e-3

python3 src/train_decoder.py \
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
    --pos_mse_weight 0 \
    --auxiliary_kl_weight 0 \
    --subset_ratio 0.01 \
    --default_save_path "$save_path" \
    --exp_name phase_3 \
    --gpus 1 \
    --prior_type "optimized_data" \
    --decoder_checkpoint "$decoder_checkpoint" \
    
    