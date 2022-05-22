# Adapted from 3/27/22: https://github.com/shijie-wu/crosslingual-nlp/blob/master/src/train.py

# The argument to run the general grid search script
# ./example/surprising-mbert/evaluate.sh 


# The argument to actually use
# python3 scratchwork.py --task=udpos --data_dir=/nobackup/users/wongn/ud-treebanks-v1.4 --trn_langs=English --val_langs=English --tst_langs="English Spanish" --pretrain=bert-base-multilingual-cased

# the argument to train
# python3 ./src/train.py --task=udpos --data_dir=/nobackup/users/wongn/ud-treebanks-v1.4 --trn_langs=English --val_langs=English --tst_langs="English Spanish" --pretrain=bert-base-multilingual-cased


import os
from argparse import ArgumentParser

import pytorch_lightning as pl

import util
from enumeration import Task
from model import Aligner, Classifier, DependencyParser, Model, Tagger

def main(hparams):
    
    model = Tagger(hparams)
    print(len(model.train_dataloader()))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default="default", type=str)
    parser.add_argument("--min_delta", default=1e-3, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--save_last", default=False, type=util.str2bool)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--do_train", default=True, type=util.str2bool)
    parser.add_argument("--do_test", default=True, type=util.str2bool)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--cache_dataset", default=False, type=util.str2bool)
    parser.add_argument("--cache_path", default="", type=str)
    ############################################################################
    parser.add_argument("--default_save_path", default="./", type=str)
    parser.add_argument("--gradient_clip_val", default=0, type=float)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--gpus", default=None, type=int)
    parser.add_argument("--overfit_batches", default=0.0, type=float)
    parser.add_argument("--track_grad_norm", default=-1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=util.str2bool)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--min_steps", default=None, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--log_every_n_steps", default=10, type=int)
    parser.add_argument("--accelerator", default=None, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--amp_backend", default="native", type=str)
    # only used for non-native amp
    parser.add_argument("--amp_level", default="01", type=str)
    ############################################################################
    parser = Model.add_model_specific_args(parser)
    parser = Tagger.add_model_specific_args(parser)
    parser = Classifier.add_model_specific_args(parser)
    parser = DependencyParser.add_model_specific_args(parser)
    parser = Aligner.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)