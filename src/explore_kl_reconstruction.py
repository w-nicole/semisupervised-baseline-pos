
import os
import glob
import torch
import matplotlib.pyplot as plt

import explore_predict
from model import VAE
from enumeration import Split
from metric import LABEL_PAD_ID
import util

def get_spaced_out_checkpoints(model_folder):
    
    checkpoints_folder = os.path.join(model_folder, 'ckpts')
    all_checkpoint_paths = glob.glob(checkpoints_folder + '/*')
    
    step_between_checkpoints = 3
    start_from = 1 # chosen manually so that best checkpoint would be loaded
    
    chosen_checkpoint_paths = []
    for i in range(start_from, len(all_checkpoint_paths), step_between_checkpoints):
       matched = glob.glob(os.path.join(checkpoints_folder + f'/ckpts_epoch={i}'))
       assert len(matched) == 1, matched
       chosen_checkpoint_paths.extend(matched)
    
def reconstruct_kl_over_checkpoints():
    """Incomplete"""
    for checkpoint_path in chosen_checkpoint_paths:
        model = VAE.load_from_checkpoint(checkpoint_path)
        current_predictions = explore_predict.get_all_predictions(model, langs)
        current_predictions
    
def calculate_kl_tensor_against_prior(log_q_given_input, prior):
    repeated_prior = util.apply_gpu(prior.unsqueeze(0).repeat(log_q_given_input.shape[0], 1))
    pre_sum = torch.exp(log_q_given_input) * (log_q_given_input - torch.log(repeated_prior))
    assert pre_sum.shape == log_q_given_input.shape, f'pre_sum: {pre_sum.shape}, q_given_input: {log_q_given_input.shape}'
    return pre_sum
    
def reconstruct_kl_tensor(model, lang, predictions_dict, analysis_path):
    
    log_q_given_input = explore_predict.get_log_q_given_input(predictions_dict, lang)
    
    model_prior = model.prior_param.prior
    english_prior = model.get_smoothed_english_prior()
    
    kl_tensor_model = calculate_kl_tensor_against_prior(log_q_given_input, model_prior)
    kl_tensor_english = calculate_kl_tensor_against_prior(log_q_given_input, english_prior)
    
    print('kl tensor model', kl_tensor_model.shape)
    
    tensor_path = os.path.join(analysis_path, f'{lang}_kl_tensors.pt')
    results = {
        'model' : kl_tensor_model,
        'english' : kl_tensor_english,
        'model_prior' : model_prior,
        'english_prior' : english_prior
    }
    torch.save(results, tensor_path)
    print(f'Written KL tensor reconstruction to {tensor_path}')
    return results
    
def save_histogram_kl_examples(kl_tensor_results, lang, analysis_path):
    
    key = ['model', 'english']
    for prior_name, color in zip(key, ['g', 'r']):
        kl_tensor = kl_tensor_results[prior_name]
        kl_divergence = torch.sum(kl_tensor, axis = -1)
        
        plt.hist(
            util.remove_from_gpu(kl_divergence).numpy().flat,
            label = prior_name, alpha = 0.5, color = color
        )
    
    plt.title(f'KL of {lang} examples. Mean: {kl_divergence.mean()}')
    plt.xlabel('KL')
    plt.ylabel("Frequency")
    plt.legend()
    
    figure_path = os.path.join(analysis_path, f'{lang}_kl_examples.png')
    plt.savefig(figure_path)
    
    print(f'Written to {figure_path}')
    
if __name__ == '__main__':
    
    lang = 'Dutch'
    model_folder = './experiments/decoder_baseline/one_kl_weight_with_ref_val'
    checkpoint_path = os.path.join(model_folder, 'ckpts', 'ckpts_epoch=1-val_acc=81.349.ckpt')    
    
    model = VAE.load_from_checkpoint(checkpoint_path)
    analysis_path = explore_predict.get_analysis_path(checkpoint_path)
    
    prediction_path = os.path.join(analysis_path, 'predictions.pt')
    if os.path.exists(prediction_path):
        predictions_dict = torch.load(prediction_path)
    else:
        predictions_dict = explore_predict.get_all_predictions(model, [lang])
        
    kl_path = os.path.join(analysis_path, f'{lang}_kl_tensors.pt')
    # if os.path.exists(kl_path):
    #     all_kl = torch.load(kl_path)
    # else:
    #     all_kl = reconstruct_kl_tensor(model, lang, predictions_dict, analysis_path)
    # save_histogram_kl_examples(all_kl, lang, analysis_path)
    
    
    


        
        
        
