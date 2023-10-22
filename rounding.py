import torch
import sys, yaml, os
import json
import numpy as np

def get_knn(model_emb:torch.Tensor, text_emb:torch.Tensor, dist='cos'):
    """
    model_emb -> [V X d]
    text_emb -> [B X seq_len X d]
    """
    if dist == 'cos':
        sim = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)
    elif dist == 'l2':
        diff = model_emb.unsqueeze(1) - text_emb.unsqueeze(0).expand(model_emb.size(0), -1, -1)
        sim = -torch.norm(diff, dim=-1)  # used negative sign since L2 is a distance measure
    topk_output = torch.topk(sim, k=6, dim=0)
    return topk_output.values, topk_output.indices

def get_knn_efficient(model_emb:torch.Tensor, text_emb:torch.Tensor):
    """
    model_emb -> [V X d]
    text_emb -> [B X seq_len X d]
    """
    model_emb_norm = (model_emb ** 2).sum(-1).view(-1, 1)   # [V, 1]
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # [d X B*seq_len]
    text_emb_norm = (text_emb ** 2).sum(-1).view(-1, 1) # [B*seq_len, 1]
    dist = (model_emb_norm + 
            text_emb_norm.transpose(0, 1) - 
            2.0 * torch.mm(model_emb, text_emb_t)) # [V X B*seq_len]
    dist = torch.clamp(dist, 0.0, np.inf)
    topk_output = torch.topk(-dist, k=1, dim=0)
    return topk_output.values, topk_output.indices

def rounding_function(text_emb_list, model, tokenizer, emb_scale_factor=1.0):
    pass