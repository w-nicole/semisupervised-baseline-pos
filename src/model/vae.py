
# Referenced/modified code from Shijie Wu's crosslingual-nlp repository,
#   particularly `model/tagger.py`.
# Taken code includes basic class structure, imports, and method headers,
# such as initialization code.


import torch.nn.functional as F
import yaml
from argparse import Namespace
import os
import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from metric import LABEL_PAD_ID

from model import BaseVAE
import constant
import util

class VAE(BaseVAE):
    
    def __init__(self, hparams):
        super(VAE, self).__init__(hparams)
        decoder = BaseVAE.load_from_checkpoint(self.hparams.decoder_checkpoint)
        
        prior = self.get_smoothed_english_prior()
        self.prior_param = torch.nn.ParameterDict()
        self.prior_param['prior'] = torch.nn.Parameter(prior, requires_grad = True)
        self._selection_criterion = 'val_acc'
        self._comparison_mode = 'max'
        
                
    def calculate_log_pi_t(self, batch, hs):
        logits = self.classifier(hs)
        raw_log_pi_t = F.log_softmax(logits, dim=-1)
        # Need to remove the log_pi_t that do not correspond to real inputs
        log_pi_t = self.set_padded_to_zero(batch, raw_log_pi_t)
        return log_pi_t
        
    def forward(self, batch):
        
        current_language = batch['lang'][0]
        assert not any(list(filter(lambda example : example != current_language, batch['lang'])))
        
        # Labeled case
        if current_language == constant.SUPERVISED_LANGUAGE:
            return BaseVAE.forward(self, batch)
            
        # Unlabeled case
        hs = self.calculate_hidden_states(batch)
        log_pi_t = self.calculate_log_pi_t(batch, hs)
        # Calculate predicted mean
        uniform_sample = Uniform(torch.zeros(log_pi_t.shape), torch.ones(log_pi_t.shape)).rsample()
        noise = util.apply_gpu(-torch.log(-torch.log(uniform_sample)))
        
        unnormalized_pi_tilde_t = (log_pi_t + noise) / self.hparams.temperature
        pi_tilde_t = F.softmax(unnormalized_pi_tilde_t, dim=-1)
        mu_t = self.calculate_decoder(pi_tilde_t)

        loss = self.calculate_agnostic_loss(batch, mu_t, hs)
        with torch.no_grad():
            loss['encoder_loss'] = F.nll_loss(
                    log_pi_t.view(-1, self.nb_labels),
                    batch["labels"].view(-1),
                    ignore_index=LABEL_PAD_ID,
            )
        # KL calculation
        log_q_given_input = log_pi_t.sum(dim=1) 
            
        repeated_prior = self.prior_param.prior.unsqueeze(0).repeat(log_q_given_input.shape[0], 1)
        
        pre_sum = torch.exp(log_q_given_input) * (log_q_given_input - torch.log(repeated_prior))
        assert pre_sum.shape == log_q_given_input.shape, f'pre_sum: {pre_sum.shape}, q_given_input: {log_q_given_input.shape}'
        
        kl_divergence = torch.sum(pre_sum, axis = -1)

        loss['KL'] = kl_divergence.mean()
        loss['decoder_loss'] = loss['MSE'] + loss['KL']        
        
        return loss, log_pi_t
    
    def train_dataloader(self):
        assert not self.hparams.mix_sampling, "This must be set to false for the batches to work."
        assert any([current_dataset.lang == constant.SUPERVISED_LANGUAGE for current_dataset in self.trn_datasets])
        return super().train_dataloader()
        
    def evaluation_step_helper(self, batch, prefix):
        
        loss, encoder_log_probs = self.forward(batch)
        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language"
        lang = batch["lang"][0]
        self.metrics[lang].add(batch["labels"], encoder_log_probs)
        return loss
        
        