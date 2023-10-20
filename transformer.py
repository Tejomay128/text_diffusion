from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerModel(nn.Module):
    """
    Transformer model class with an LM head same as the embedding matrix which maps
    dicrete tokens to continuous space.
    """
    def __init__(self,
                 input_dims,
                 output_dims,
                 hidden_t_dim,
                 dropout=0,
                 config=None,
                 config_name="bert-base-uncased",
                 vocab_size=None,
                 init_pretrained="no",
                 logits_mode=1) -> None:
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_t_dim = hidden_t_dim
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        """
        word_embedding -> maps discrete tokens to continuous embeddings
        lm_head -> maps output embeddings back to vocabulary (tokens)
        """

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims) 
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        #weight sharing
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight  #lm-head and word embeddings are shared
        
        time_embed_dim = hidden_t_dim * 4

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_t_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, self.hidden_size)
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), 
                                              nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print("Using pretrained BERT weights.....")
            print(config)
            lm = BertModel.from_pretrained(config_name, config=config)
            self.word_embedding = lm.embeddings.word_embeddings  # set word embeddings to pretrained BERT embeddings

            with torch.no_grad():
                self.lm_head.weight = self.word_embedding.weight  # lm-head and word embeddings are shared
            
            self.encoder = lm.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = lm.embeddings.position_embeddings
            self.LayerNorm = lm.embeddings.LayerNorm

            del lm.embeddings
            del lm.pooler
        
        elif init_pretrained == 'no':
            self.encoder = BertEncoder(config=config)
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "Pretrained type not supported."
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                  nn.Tanh(), 
                                                  nn.Linear(config.hidden_size, self.output_dims))
        
    def get_embeddings_from_input_ids(self, input_ids):
        return self.word_embedding(input_ids)
        
    def get_logits(self, hidden_repr):
        """
        Given output hidden representations, obtain logits over the vocabulary
        """
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def forward(self, x, timesteps):
        """
        Apply the transformer model to an input batch.

        -> x: an [N x seq_len x ...] Tensor of inputs.
        -> timesteps: a 1-D batch of timesteps.
        :return: an [N x seq_len x ...] Tensor of outputs.
        """
        