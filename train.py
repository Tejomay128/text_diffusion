import torch
from transformers import set_seed

import json, torch, os, argparse
import numpy as np

from step_sampler import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer,
    create_model_and_diffusion,
    args_to_dict
)

from datasets_utils import load_data_text

device = torch.device("cuda:0")

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)

    tokenizer = load_tokenizer(args)
    embs, tokenizer = load_model_emb(args, tokenizer)

    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb = embs # use model's weights as init
    )

    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=embs # using the same embedding wight with tranining data
    )

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, load_defaults_config().keys()))
    model.to(device)

    timstep_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    

