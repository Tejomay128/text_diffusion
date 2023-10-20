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
            self.lm_head.weight = self.word_embedding.weight
        
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