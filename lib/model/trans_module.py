'''
This file is modified from:
https://github.com/rishikksh20/CrossViT-pytorch/blob/master/crossvit.py
'''

import torch
from torch import nn, einsum
import torch.nn.functional as F

import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)

class FeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., apply_transform=False, transform_scale=True, knn_attention=0.7):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.apply_transform = apply_transform
        self.knn_attention = bool(knn_attention)
        self.topk = knn_attention

        if apply_transform:
            self.reatten_matrix = torch.nn.Conv2d(heads, heads, 1, 1)
            self.var_norm = torch.nn.BatchNorm2d(heads)
            self.reatten_scale = self.scale if transform_scale else 1.0

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.scores = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.knn_attention:
            mask = torch.zeros(b, self.heads, n, n, device=x.device, requires_grad=False)
            index = torch.topk(dots, k=int(dots.size(-1)*self.topk), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            dots = torch.where(mask > 0, dots, torch.full_like(dots, float('-inf')))
        attn = dots.softmax(dim=-1)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale

        self.scores = attn
        out = einsum('b h i j, b h j d -> b h i d', attn, v)


        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
