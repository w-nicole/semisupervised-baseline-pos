
import os
import torch
import numpy as np
from transformers import BertTokenizer
import pandas as pd
import itertools

from model import Single
import util
from enumeration import Split
import constant
from constant import LABEL_PAD_ID
import predict.match_filtering as match_filtering
import predict.predict_utils as predict_utils


def get_mask_index(masked_example, tokenizer):
    matching_indices = np.where(masked_example == tokenizer.convert_tokens_to_ids(tokenizer.mask_token))[0]
    assert matching_indices.shape[0] == 1
    return matching_indices.item()
    
def token_to_word(tokens, tokenizer):
    return ' '.join(filter(lambda token : token != tokenizer.mask_token, tokenizer.convert_ids_to_tokens(tokens)))
    
def generate_sentence_df(right_args, wrong_args, is_right_masked, tokenizer):
    unmasked_args, masked_args = (wrong_args, right_args) if is_right_masked else (right_args, wrong_args)
    dense_max_softmaxes, dense_predictions = {}, {}
    # Get dense right and wrong predictions, softmaxes.
    for sweep_name, args in zip(['right', 'wrong'], [right_args, wrong_args]):
        dense_softmax = match_filtering.clean_for_ensemble_softmax(args['softmax'], args['flat_labels']).cpu()
        dense_predictions[sweep_name] = dense_softmax.argmax(dim=-1)
        dense_max_softmaxes[sweep_name] = dense_softmax.max(dim=-1)[0]
        
    clean_labels = lambda flat_labels : flat_labels[flat_labels != LABEL_PAD_ID]
    assert torch.all(clean_labels(right_args['flat_labels']) == clean_labels(wrong_args['flat_labels']))
    dense_labels = clean_labels(right_args['flat_labels'])
    
    assert dense_predictions['right'].shape == dense_predictions['wrong'].shape
    mask = (dense_predictions['right'] == dense_labels) & (dense_predictions['wrong'] != dense_labels)
    
    all_prediction_related_tensors = [
        tensor
        for tensor_group in [dense_predictions, dense_max_softmaxes]
        for tensor in tensor_group.values()
    ]
    assert all(map(lambda tensor : len(tensor.shape) == 1, all_prediction_related_tensors))
    assert len(mask.shape) == 1
    
    masked_predictions = { name : prediction[mask] for name, prediction in dense_predictions.items() }
    masked_max_softmaxes = { name : max_softmaxes[mask] for name, max_softmaxes in dense_max_softmaxes.items() }
    masked_labels = dense_labels[mask]
        
    unmasked_dataloader = unmasked_args['dataloader']
    masked_dataloader = masked_args['dataloader']
    # Prep for softmax value and token extraction.
    unmasked_labels_by_dataloader = list(map(lambda example : example['pos_labels'].cpu(), unmasked_dataloader))
    labels_per_example = [
            (labels != LABEL_PAD_ID).sum().item()
            for label_batch in unmasked_labels_by_dataloader
            for labels in label_batch 
        ]
    unmasked_examples = list(itertools.chain(*list(map(lambda example : list(example['sent'].cpu()), unmasked_dataloader))))
    raw_masked_examples = list(itertools.chain(*list(map(lambda example : list(example['sent'].cpu()), masked_dataloader))))
    assert len(labels_per_example) == len(unmasked_examples)
    repeated_unmasked_examples = [
            torch.tile(sentence.unsqueeze(0), dims = (current_label_count, 1)).numpy()
            for sentence, current_label_count in zip(unmasked_examples, labels_per_example)
        ]
    flattened_unmasked_examples = list(itertools.chain(*repeated_unmasked_examples))
    dense_unmasked_examples_list = util.pad_batch(
            flattened_unmasked_examples,
            padding = int(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
        )
    
    dense_unmasked_examples = torch.from_numpy(np.stack(dense_unmasked_examples_list, axis = 0))
    bert_mask_indices = np.array(list(map(lambda example : get_mask_index(example, tokenizer), raw_masked_examples)))
    dense_tokens = np.array(list(map(lambda args : args[0][args[1]].item(), list(zip(dense_unmasked_examples, bert_mask_indices)))))
    masked_examples = np.take(dense_unmasked_examples.numpy(), np.where(mask.numpy())[0], axis = 0)
    masked_tokens = dense_tokens[mask] 
    
    all_first_dims = {
        tensor.shape[0]
        for tensor in [ dense_tokens, dense_labels, dense_unmasked_examples ] + all_prediction_related_tensors
    }
    
    if not len(all_first_dims) == 1: import pdb; pdb.set_trace()
    map_label_to_word = lambda labels : list(map(lambda label : constant.UD_POS_LABELS[label], labels))

    df_tokens = [ token_to_word([token], tokenizer) for token in masked_tokens ]
    df_labels = map_label_to_word(masked_labels)
    df_right_prediction = map_label_to_word(masked_predictions['right'])
    df_wrong_prediction = map_label_to_word(masked_predictions['wrong'])
    df_context = [ token_to_word(example, tokenizer) for example in masked_examples ]
    df = pd.DataFrame.from_records({
        'token' : df_tokens,
        'label' : df_labels,
        'right_prediction' : df_right_prediction,
        'wrong_prediction' : df_wrong_prediction,
        'right_max_softmax' : masked_max_softmaxes['right'],
        'wrong_max_softmax' : masked_max_softmaxes['wrong'],
        'context' : df_context
    })
    column_order = ['token', 'label', 'right_prediction', 'wrong_prediction', 'right_max_softmax', 'wrong_max_softmax', 'context']
    df = df[column_order].sort_values(by=['right_max_softmax'])

    return df
    
if __name__ == '__main__':
    
    lang = 'English'
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    subset = 10
    base_path = '../../alt/semisupervised-baseline-pos/'
    sweep_path = os.path.join(base_path, 'experiments/self_train/english')
    comparison_path = os.path.join(sweep_path, f'cross_data/subset_count={subset}')
    
    loading_models = { is_masked : util.get_subset_model(Single, is_masked) for is_masked in [True, False] }
    
    base_folder = '../../alt/semisupervised-baseline-pos/experiments/self_train'
    masked_folder = 'english/mixed/mixed/version_38g602fp/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=98.615.ckpt'
    unmasked_folder = 'english/pure/pure/version_295owqwl/ckpts/ckpts_epoch=3-val_English_pos_acc_epoch=98.047.ckpt'
    
    flip_labels = True
    softmax_path = f'flipped_true_labels/val_predictions/{lang}/dev_predictions.pt'
    
    masked_args = {
        'softmax' : torch.load(os.path.join(sweep_path, masked_folder, softmax_path))[lang],
        'flat_labels' : predict_utils.get_batch_padded_flat_labels(loading_models[True], lang, Split.dev).cpu(),
        'dataloader' : util.get_subet_dataloader(loading_models[False if flip_labels else True], lang, Split.dev)
    }
    unmasked_args = {
        'softmax' : torch.load(os.path.join(sweep_path, unmasked_folder, softmax_path))[lang],
        'flat_labels' : predict_utils.get_batch_padded_flat_labels(loading_models[False], lang, Split.dev).cpu(),
        'dataloader' : util.get_subset_dataloader(loading_models[True if flip_labels else False], lang, Split.dev)
    }
    masked_right_df = generate_sentence_df(masked_args, unmasked_args, True, tokenizer)
    unmasked_right_df = generate_sentence_df(unmasked_args, masked_args, False, tokenizer)
    for (right_name, wrong_name), df in zip([('masked', 'unmasked'), ('unmasked', 'masked')], [masked_right_df, unmasked_right_df]):    
        analysis_folder = os.path.join(comparison_path, 'misclassified_sentences')
        if not os.path.exists(analysis_folder): os.makedirs(analysis_folder)
        df_path = os.path.join(analysis_folder, f'right={right_name}_wrong={wrong_name}_sorted_by_right.csv')
        df.to_csv(df_path)