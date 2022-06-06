
# Referenced/modified `model/tagger.py` from Shijie Wu's crosslingual-nlp repository.
# Taken code includes basic class structure, imports, and method headers, general method logic excluding new functionality/irrelevant old functionality
#   as well as inheritance-related code (such as the declaration of the argparse)
#   or code that directly overrides previous code
#  (like the mix_sampling default value,
#  the logic of the forward code was maintained with irrelevant code removed, with new logic added)  
#  Most other code was added.
# See LICENSE in this codebase for license information.

import torch

from model.tagger import Tagger
import util

class EncoderDecoder(Tagger):
    
    def __init__(self, hparams):
        super(EncoderDecoder, self).__init__(hparams)
        weights = torch.load(hparams.encoder_checkpoint, **{'map_location' : torch.device('cpu') } if not torch.cuda.is_available() else {})
        self.classifier = Tagger.load_weights(weights)
        self.decoder = torch.nn.LSTM(
            input_size = self.hidden_size, # Encoder input size.
            hidden_size = hparams.decoder_hidden_size,
            num_layers = hparams.decoder_number_of_layers,
            batch_first = True,
            bidirectional = True
        )
        english_train_prior = self.get_english_train_prior()
        param = torch.nn.Parameter(english_train_prior, requires_grad = True)
        
    # Shijie Wu's code, but with decoder logic added and irrelevant options removed,
    # and variables renamed for notation consistency.
    def forward(self, batch):
        batch = self.preprocess_batch(batch)
        # Updated call arguments
        hs = self.encode_sent(batch["sent"], batch["start_indices"], batch["end_indices"], batch["lang"])
        logits = self.classifier(hs)
        log_pi_t = F.log_softmax(logits, dim=-1)
        
        # Calculate predicted mean
        uniform_sample = Uniform(torch.zeros(log_pi_t.shape), torch.ones(log_pi_t.shape)).rsample()
        noise = util.apply_gpu(-torch.log(-torch.log(uniform_sample)))
        
        unnormalized_pi_tilde_t = (log_pi_t + noise) / hparams.temperature
        pi_tilde_t = F.softmax(unnormalized_pi_tilde_t, dim=-1)
        
        mu_t = self.decoder(pi_tilde_t)
        
        # Calculate losses
        loss = {}
        loss['encoder_loss'] = F.nll_loss(
                pi_t.view(-1, self.nb_labels),
                batch["labels"].view(-1),
                ignore_index=LABEL_PAD_ID,
        )
        pi_t = torch.exp(log_pi_t)
        assert len(pi_t.shape) == 3, pi_t.shape
        
        q_given_input = torch.ones((pi_t.shape[0], pi_t.shape[-1]))
        for index in pi_t.shape[1]:
            q_given_input *= pi_t[:, index, :]
        
        pre_sum = q_given_input * torch.log(q_given_input / self.english_train_prior)
        assert pre_sum.shape == q_given_input.shape, f'pre_sum: {pre_sum.shape}, q_given_input: {q_given_input.shape}'
        kl_divergence = torch.sum(pre_sum, axis = -1)
        
        loss['KL'] = kl_divergence
        loss['MSE'] = F.mse_loss(mu_t, hs)
        loss['decoder_loss'] = loss['KL'] + loss['MSE']
        
        return loss, log_pi_t
        
    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--temperature", default=1, type=float)
        parser.add_argument("--decoder_hidden_size", default=1, type=int)
        parser.add_argument("--decoder_number_of_layers", default=1, type=int)
        return parser
    
    def get_english_train_prior(self):
        matching_indices = np.where(np.array(hparams.trn_langs) == 'English')[0]
        assert matching_indices.shape[0] == 1 and len(matching_indices.shape) == 1, f'Actual indices: {matching_indices}'
        
        language_index = matching_indices.item()
        
        english_dataset = self.trn_datasets[language_index]
        assert english_dataset.lang == 'English', f'Actual language: {english_dataset.lang}'
        
        train_data = english_dataset.read_file(english_dataset.filepath, english_dataset.lang, english_dataset.split)
        
        labels = []
        for data in train_data:
            labels.extend(data['labels'])
            
        numerical_labels = list(map(lambda label : english_dataset.label2id[label], labels))
        return torch.bincount(numerical_labels, minlength = english_dataset.nb_labels)
    
    def evaluation_step_helper(self, batch, prefix):
        loss, encoder_log_probs = self.forward(batch)
        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language"
        lang = batch["lang"][0]
        self.metrics[lang].add(batch["labels"], log_probs)
        return loss
            
        
        
        
        
        
    
