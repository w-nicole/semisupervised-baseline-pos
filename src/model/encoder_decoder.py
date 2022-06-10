
# Referenced/modified code from Shijie Wu's crosslingual-nlp repository,
#   particularly `model/tagger.py`.
# Taken code includes basic class structure, imports, and method headers,
# general method logic for all methods that also exist in `tagger.py`,
#   excluding new functionality per new decoder logic/irrelevant old functionality
#   as well as inheritance-related code (such as the declaration of the argparse)
#   or code that directly overrides previous code
#  (like the mix_sampling default value,
#  the logic of the forward/training_step code was maintained with irrelevant code removed, with new logic added)  
#  Most other code was added.
# See LICENSE in this codebase for license information.

import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
import numpy as np

from dataset import LABEL_PAD_ID
from dataset import tagging
from model.tagger import Tagger
from enumeration import Split
import util
import metric
import constant

class EncoderDecoder(Tagger):
    
    def __init__(self, hparams):
        super(EncoderDecoder, self).__init__(hparams)
        encoder = Tagger.load_from_checkpoint(self.hparams.encoder_checkpoint)
        self.freeze_bert(encoder)
        # Overwrite base and tagger attributes so that encode_sent will function correctly
        self.model = encoder.model
        self.classifier = encoder.classifier
        self.decoder_lstm = torch.nn.LSTM(
                input_size = len(constant.UD_POS_LABELS),
                hidden_size = self.hidden_size, # Encoder input size
                num_layers = self.hparams.decoder_number_of_layers,
                batch_first = True,
                bidirectional = True
        )
        self.decoder_linear = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        prior = self.get_uniform_prior()
        self.prior_param = torch.nn.ParameterDict()
        self.prior_param['prior'] = torch.nn.Parameter(prior, requires_grad = True)
        
    def freeze_bert(self, encoder):
        # Adapted from model/base.py by taking the logic to freeze up to and including a certain layer
        # Doesn't freeze the pooler, but encode_sent excludes pooler correctly.
        encoder.freeze_embeddings()
        for index in range(encoder.model.config.num_hidden_layers + 1):
            encoder.freeze_layer(index)
        # end adapted
        
    # Shijie Wu's code, but with decoder logic added and irrelevant options removed,
    # and variables renamed for notation consistency.
    def forward(self, batch):
        
        batch = self.preprocess_batch(batch)
        # Updated call arguments
        hs = self.encode_sent(batch["sent"], batch["start_indices"], batch["end_indices"], batch["lang"])
        logits = self.classifier(hs)
        raw_log_pi_t = F.log_softmax(logits, dim=-1)
        # Need to remove the log_pi_t that do not correspond to real inputs
        repeated_labels = batch['labels'].unsqueeze(2).repeat(1, 1, raw_log_pi_t.shape[-1])
        log_pi_t = torch.where(repeated_labels != metric.LABEL_PAD_ID, raw_log_pi_t, torch.zeros(raw_log_pi_t.shape).cuda())

        # Calculate predicted mean
        uniform_sample = Uniform(torch.zeros(log_pi_t.shape), torch.ones(log_pi_t.shape)).rsample()
        noise = util.apply_gpu(-torch.log(-torch.log(uniform_sample)))
        
        unnormalized_pi_tilde_t = (log_pi_t + noise) / self.hparams.temperature
        pi_tilde_t = F.softmax(unnormalized_pi_tilde_t, dim=-1)
        
        # "Reshape" the bidirectional concatenation
        mu_t_raw, _ = self.decoder_lstm(pi_tilde_t)
        mu_t = self.decoder_linear(mu_t_raw)
        
        # Calculate losses
        loss = {}
        with torch.no_grad():
            loss['encoder_loss'] = F.nll_loss(
                    log_pi_t.view(-1, self.nb_labels),
                    batch["labels"].view(-1),
                    ignore_index=LABEL_PAD_ID,
            )

        log_q_given_input = util.apply_gpu(torch.zeros((log_pi_t.shape[0], log_pi_t.shape[-1])))
        log_q_given_input = log_pi_t.sum(dim=1)
            
        repeated_prior = self.prior_param.prior.unsqueeze(0).repeat(log_q_given_input.shape[0], 1)
        
        pre_sum = torch.exp(log_q_given_input) * (log_q_given_input - torch.log(repeated_prior))
        assert pre_sum.shape == log_q_given_input.shape, f'pre_sum: {pre_sum.shape}, q_given_input: {log_q_given_input.shape}'
        kl_divergence = torch.sum(pre_sum, axis = -1)
        
        loss['KL'] = kl_divergence.mean()
        
        loss['MSE'] = F.mse_loss(mu_t, hs)
        loss['decoder_loss'] = -(loss['MSE'] - loss['KL'])

        if loss['KL'] == 0: import pdb; pdb.set_trace()
        return loss, log_pi_t
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--temperature", default=1, type=float)
        parser.add_argument("--decoder_number_of_layers", default=1, type=int)
        return parser
    
    def get_uniform_prior(self):
        number_of_labels = len(constant.UD_POS_LABELS)
        return torch.ones(number_of_labels) / number_of_labels
        
    def get_english_prior(self):
        counts = self.get_label_counts('English', Split.train)
        return counts / torch.sum(counts)
    
    def training_step(self, batch, batch_idx):
        loss_dict, _ = self.forward(batch)
        loss = loss_dict['decoder_loss']
        self.log("loss", loss)
        
        try:
            return loss
        except:
            import pdb
            pdb.set_trace()
    
    def evaluation_step_helper(self, batch, prefix):
        loss, encoder_log_probs = self.forward(batch)
        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language"
        lang = batch["lang"][0]
        self.metrics[lang].add(batch["labels"], encoder_log_probs)
        return loss
    