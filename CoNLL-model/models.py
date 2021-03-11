"""
Module for models storing.

"""

import torch
from torch import nn
from transformers import BertForTokenClassification

from bert_config import *


class BEboC(nn.Module):
    """
    BERT+Elmo+biLSTM+one CRF
    """
    def __init__(self, hidden_size=128, num_labels=4, elmo_layers=2,
                 bert_layers=1, concat_bert=True, bilstm_layers=1):
        """
        Creates model
        
        Parameters
        ----------
        hidden_size:
        num_labels:
        elmo_layers: int, default=2
            Num of ELMo layers to be considered
        bert_layers: int, default=1
            Num of final BERT hidden layers to be used as embedding vector.
        concat_bert: bool, default=True
            Whether to concat (True) or sum (False) last BERT hidden layers.
        bilstm_layers: int, default=1
        """
        super(BEboC, self).__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.elmo_layers = elmo_layers
        self.bert_layers = bert_layers
        self.concat_bert = concat_bert
        self.bilstm_layers = bilstm_layers
        
        self.bert = BertForTokenClassification.from_pretrained(
                        BERT_MODEL,
                        output_hidden_states=True)
        
        for pars in self.bert.parameters():
            pars.requires_grad = False
        
        bert_embedding_dim = self.bert.config.to_dict()['hidden_size']

        self.elmo = Elmo(options_file, weight_file, self.elmo_layers, dropout=0, requires_grad=False)
        
        elmo_embedding_dim = 512 # it's always fixed

        if self.concat_bert:
          self.linear1 = nn.Linear(bert_embedding_dim*self.bert_layers+elmo_embedding_dim*self.elmo_layers, 1024)
        else:
          self.linear1 = nn.Linear(bert_embedding_dim+elmo_embedding_dim*self.elmo_layers, 1024)
        
        self.bilstm = nn.LSTM(1024, self.hidden_size, self.bilstm_layers, bidirectional=True)
        
        self.linear2 = nn.Linear(self.hidden_size*2, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
    
    def get_model_pars_dict(self):
        """
        Returns dict with described model's parameters.
        
        """
        pars = {}
        pars['hidden_size'] = self.hidden_size
        pars['num_labels'] = self.num_labels
        pars['elmo_layers'] = self.elmo_layers
        pars['bert_layers'] = self.bert_layers
        pars['concat_bert'] = int(self.concat_bert)
        pars['bilstm_layers'] = self.bilstm_layers

        return pars
    
    def forward(self, bert_ids, elmo_ids, attention_mask):
        """
        Forward propogate of model.
        
        Parameters
        ----------
        sequence:
        attention_mask:
        
        Returns
        -------
        Logits
        
        """

        bert_hiddens = self.bert(bert_ids, attention_mask=attention_mask)[1]
        elmo_hiddens = self.elmo(elmo_ids)

        if self.concat_bert:
            bert_embedding = torch.cat(bert_hiddens[-self.bert_layers:], dim=2)#[bert_hiddens[-i] for i in range(-1, -self.bert_layers-1, -1)], dim=0)
        else:
            emb_sum = 0
            for h in bert_hiddens[-self.bert_layers:]:
                emb_sum += h
            bert_embedding = emb_sum

        elmo_bert_embeddings = torch.clone(bert_embedding)

        for el_hi in elmo_hiddens['elmo_representations']:
            elmo_bert_embeddings = torch.cat((elmo_bert_embeddings, el_hi), dim=-1)

        linear1_output = nn.functional.relu(self.linear1(elmo_bert_embeddings))

        bilstm_output, (h_n, c_n) = self.bilstm(linear1_output)
        linear2_output = nn.functional.relu(self.linear2(bilstm_output))
        return linear2_output