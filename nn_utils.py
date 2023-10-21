import math
import torch
import torch.nn as nn


def time_embedding(timesteps, dim, max_period=10000):
    """
    Creates sinusoidal embeddings for timesteps
    -> timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    -> dim: the dimension of the output.
    -> max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding