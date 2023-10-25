import torch
import sys, yaml, os
import json
import numpy as np

def get_knn(model_emb:torch.Tensor, text_emb:torch.Tensor, dist='cos'):
    """
    model_emb -> [V X d] -> Embeddings from the denoising model that correspond
    to some token.
    text_emb -> [B X seq_len X d] -> Typically, an output from a diffusion step
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
    model_emb -> [V X d] -> Embeddings from the denoising model that correspond
    to some token.
    text_emb -> [B X seq_len X d] -> Typically, an output from a diffusion step
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
    """
    Given a list of embeddings, rounds each embedding to the nearest token
    """
    decoded_out_lst = []
    model_emb = model.weight
    dist = 'l2'
    
    for text_emb in text_emb_list:
        text_emb = torch.tensor(text_emb)
        if len(text_emb.shape) > 2:
            text_emb = text_emb.view(-1, text_emb.size(-1)) # [B*seq_len X d]
        else:
            text_emb = text_emb
        val, indices = get_knn(model_emb,
                                text_emb.to(model_emb.device), dist=dist)
    
        decoded_out_lst.append(tokenizer.decode_token(indices[0]))

    return decoded_out_lst


def denoised_fn_round(args, model, text_emb, t):
    """
    Round off text embeddings obtained from diffusion model to the nearest embedding
    that maps back to a token.
    text_emb -> [B X seq_len X d] -> Typically, an output from a diffusion step
    model -> provides embeddings that map to some existing token
    """
    model_emb = model.weight  # input_embs
    old_shape = text_emb.shape
    old_device = text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1)) # [B*seq_len X d]
    else:
        text_emb = text_emb
    # val, indices = get_knn(model_emb, text_emb.to(model_emb.device), dist=dist)
    val, indices = get_knn_efficient(model_emb, text_emb.to(model_emb.device))
    rounded_tokens = indices[0]
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds