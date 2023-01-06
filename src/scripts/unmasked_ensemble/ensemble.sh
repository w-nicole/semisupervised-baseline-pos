
#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/unmasked_ensemble/ensemble"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}
checkpoint_1=${3:-"./experiments/unmasked_ensemble/unmasked/unmasked_subset_count=10/version_26abg0b3/ckpts/ckpts_epoch=15-val_English_pos_acc_epoch=81.096.ckpt"}
checkpoint_2=${4:-"./experiments/unmasked_ensemble/unmasked/unmasked_alt_seedsubset_count=10/version_1tkbu7av/ckpts/ckpts_epoch=11-val_English_pos_acc_epoch=79.605.ckpt"}

bs=32
ep=4

python3 src/train_on_split_ensemble.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_learning_rate 5e-3 \
    --pseudolabel_checkpoint $checkpoint_1 \
    --view_checkpoint_1 $checkpoint_1 \
    --view_checkpoint_2 $checkpoint_2 \
    --mask_probability 0 \
    --view_mask_probability_1 0 \
    --view_mask_probability_2 0 \
    --group "random_mask" \
    --job_type "unmasked_ensemble" \
    --name "ensemble" \
    --default_save_path "$save_path" \
    --schedule "reduceOnPlateau" \
    --gpus 1