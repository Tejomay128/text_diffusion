import torch
from transformers import set_seed

import json, torch, os, argparse
import numpy as np

from step_sampler import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)

from datasets_utils import load_data_text

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

