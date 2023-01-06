
#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/random_mask/ensemble"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}
checkpoint=${3:-"./experiments/random_mask/english/individual/mask_probability=0.0/version_ws2uktvc/ckpts/ckpts_epoch=15-val_English_pos_acc_epoch=81.096.ckpt"}

bs=32
ep=4
mask_probability=0

python3 src/train_on_split_ensemble.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_learning_rate 5e-3 \
    --pseudolabel_checkpoint $checkpoint \
    --view_checkpoint_1 $checkpoint \
    --view_checkpoint_2 $checkpoint \
    --mask_probability 0 \
    --view_mask_probability_1 $mask_probability \
    --view_mask_probability_2 $mask_probability \
    --group "random_mask" \
    --job_type "ensemble" \
    --name "$mask_probability" \
    --default_save_path "$save_path" \
    --schedule "reduceOnPlateau" \
    --gpus 1