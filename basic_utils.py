import argparse
import torch
import json, os
import time

import diffusion
from diffusion import SpacedDiffusion, space_timesteps
from transformer import TransformerNetModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

class myTokenizer():
    """
    Load tokenizer from bert config
    """
    ##################################################
    ### You can customize your own tokenizer here. ###
    ##################################################
    def __init__(self, args):
        if args.vocab == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(args.config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            # save
            tokenizer.save_pretrained(args.checkpoint_path)
        
        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size # update vocab size in args
    
    def encode_token(self, sentences):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "Invalid type of tokenizer"
        return input_ids
    
    def decode_token(self, seq):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id: # remove all pad tokens in the end
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of tokenizer"
        return tokens
    
    
