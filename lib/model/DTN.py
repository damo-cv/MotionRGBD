'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from torch.autograd import Variable
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import init

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import random, math
from .utils import *
from .trans_module import *

np.random.seed(123)
random.seed(123)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., apply_transform=False, knn_attention=0.7):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                       apply_transform=apply_transform, knn_attention=knn_attention)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, args, small_dim=1024, small_depth=4, small_heads=8, small_dim_head=64, hidden_dim_small=768,
                 media_dim=1024, media_depth=4, media_heads=8, media_dim_head=64, hidden_dim_media=768,
                 large_dim=1024, large_depth=4, large_heads=8, large_dim_head=64, hidden_dim_large=768,
                 dropout=0., Local_flag=True):
        super().__init__()

        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head,
                                                 mlp_dim=hidden_dim_small, dropout=dropout, knn_attention=args.knn_attention)
        self.transformer_enc_media = Transformer(media_dim, media_depth, media_heads, media_dim_head,
                                                 mlp_dim=hidden_dim_media, dropout=dropout, knn_attention=args.knn_attention)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head,
                                                 mlp_dim=hidden_dim_large, dropout=dropout, knn_attention=args.knn_attention)
        if Local_flag:
            self.Mixed_small = TemporalInceptionModule(512, [160,112,224,24,64,64], 'Mixed_small')
            self.Mixed_media = TemporalInceptionModule(512, [160,112,224,24,64,64], 'Mixed_media')
            self.Mixed_large = TemporalInceptionModule(512, [160, 112, 224, 24, 64, 64], 'Mixed_large')
            self.MaxPool = MaxPool3dSamePadding(kernel_size=[3, 1, 1], stride=(1, 1, 1), padding=0)

    def forward(self, xs, xm, xl, Local_flag=False):
        # Local Modeling
        if Local_flag:
            cls_small = xs[:, 0]
            xs = self.Mixed_small(xs[:, 1:, :].permute(0, 2, 1).view(xs.size(0), xs.size(-1), -1, 1, 1))
            xs = self.MaxPool(xs)
            xs = torch.cat((cls_small.unsqueeze(1), xs.view(xs.size(0), xs.size(1), -1).permute(0, 2, 1)), dim=1)

            cls_media = xm[:, 0]
            xm = self.Mixed_media(xm[:, 1:, :].permute(0, 2, 1).view(xm.size(0), xm.size(-1), -1, 1, 1))
            xm = self.MaxPool(xm)
            xm = torch.cat((cls_media.unsqueeze(1), xm.view(xm.size(0), xm.size(1), -1).permute(0, 2, 1)), dim=1)

            cls_large = xl[:, 0]
            xl = self.Mixed_large(xl[:, 1:, :].permute(0, 2, 1).view(xl.size(0), xl.size(-1), -1, 1, 1))
            xl = self.MaxPool(xl)
            xl = torch.cat((cls_large.unsqueeze(1), xl.view(xl.size(0), xl.size(1), -1).permute(0, 2, 1)), dim=1)

        # Global Modeling
        xs = self.transformer_enc_small(xs)
        xm = self.transformer_enc_media(xm)
        xl = self.transformer_enc_large(xl)

        return xs, xm, xl


class RCMModule(nn.Module):
    def __init__(self, args, dim_head=64, method='New', merge='GAP'):
        super(RCMModule, self).__init__()
        self.merge = merge
        self.heads = args.SEHeads
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool3d = nn.AdaptiveAvgPool3d((None, 1, None))

        # Self Attention Layers
        self.q = nn.Linear(64, dim_head * self.heads, bias=False)
        self.k = nn.Linear(64, dim_head * self.heads, bias=False)
        self.scale = dim_head ** -0.5

        self.method = method
        if method == 'Ori':
            self.norm = nn.LayerNorm(128)
            self.project = nn.Sequential(
                nn.Linear(64, 512, bias=False),
                nn.GELU(),
                nn.Linear(512, 512, bias=False),
                nn.LayerNorm(512)
            )
        elif method == 'New':
            if args.dataset == 'THU':
                hidden_dim = 128
            else:
                hidden_dim = 256
            self.project = nn.Sequential(
                nn.Linear(64, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, 64, bias=False),
                nn.LayerNorm(64),
            )
            self.linear = nn.Linear(64, 512)
            # init.kaiming_uniform_(self.linear, a=math.sqrt(5))

        if self.heads > 1:
            self.mergefc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(512 * self.heads, 512, bias=False),
                nn.LayerNorm(512)
                                         )

    def forward(self, x):
        b, c, t = x.shape
        inp = x.clone()

        # Sequence (Y) direction
        xd_weight = self.project(self.avg_pool(inp.permute(0, 2, 1)).view(b, -1))
        xd_weight = torch.sigmoid(xd_weight).view(b, -1, 1)

        # Feature (X) direction
        q, k = self.q(x), self.k(x)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k])
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if self.merge == 'mean':
            dots = dots.mean(dim=2)
        elif self.merge == 'GAP':
            dots = self.avg_pool3d(dots).squeeze()

        if self.heads > 1:
            dots = dots.view(b, -1)
            dots = self.mergefc(dots)
        else:
            dots = dots.squeeze()
        y = torch.sigmoid(dots).view(b, c, 1)

        if self.method == 'Ori':
            out = x * (y.expand_as(x) + xd_weight.expand_as(x))
            visweight = xd_weight # for visualization
            return out, xd_weight, visweight

        elif self.method == 'New':
            weight = einsum('b i d, b j d -> b i j', xd_weight, y)
            out = x * weight.permute(0, 2, 1)
            visweight = weight  # for visualization
            return out, self.linear(xd_weight.squeeze()), visweight

class DTNNet(nn.Module):
    def __init__(self, args, num_classes=249, small_dim=512, media_dim=512, large_dim=512,
                 small_depth=1, media_depth=1, large_depth=1,
                 heads=8, pool='cls', dropout=0.1, emb_dropout=0.0, branch_merge='pool',
                 init: bool = False,
                 warmup_temp_epochs: int = 30):
        super().__init__()
        self.low_frames = args.low_frames
        self.media_frames = args.media_frames
        self.high_frames = args.high_frames
        self.branch_merge = branch_merge
        self._args = args
        warmup_temp, temp = map(float, args.temp)

        multi_scale_enc_depth = args.N
        num_patches_small = self.low_frames
        num_patches_media = self.media_frames
        num_patches_large = self.high_frames

        self.pos_embedding_small = nn.Parameter(torch.randn(1, num_patches_small + 1, small_dim))
        self.cls_token_small = nn.Parameter(torch.randn(1, 1, small_dim))
        self.dropout_small = nn.Dropout(emb_dropout)

        self.pos_embedding_media = nn.Parameter(torch.randn(1, num_patches_media + 1, media_dim))
        self.cls_token_media = nn.Parameter(torch.randn(1, 1, media_dim))
        self.dropout_media = nn.Dropout(emb_dropout)

        self.pos_embedding_large = nn.Parameter(torch.randn(1, num_patches_large + 1, large_dim))
        self.cls_token_large = nn.Parameter(torch.randn(1, 1, large_dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        Local_flag = True
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(
                MultiScaleTransformerEncoder(args, small_dim=small_dim, small_depth=small_depth,
                                             small_heads=heads,

                                             media_dim=media_dim, media_depth=media_depth,
                                             media_heads=heads,

                                             large_dim=large_dim, large_depth=large_depth,
                                             large_heads=heads,
                                             dropout=dropout,
                                             Local_flag=Local_flag))
            Local_flag = False
        self.pool = pool
        # self.to_latent = nn.Identity()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        if self._args.recoupling:
            self.rcm = RCMModule(args)

        if args.Network != 'FusionNet':
            self.mlp_head_small = nn.Sequential(
                nn.LayerNorm(small_dim),
                nn.Linear(small_dim, num_classes),
                # nn.Dropout(0.4)
            )
            self.mlp_head_media = nn.Sequential(
                nn.LayerNorm(media_dim),
                nn.Linear(media_dim, num_classes),
                # nn.Dropout(0.4)

            )

            self.mlp_head_large = nn.Sequential(
                nn.LayerNorm(large_dim),
                nn.Linear(large_dim, num_classes),
                # nn.Dropout(0.4)
            )

        self.show_res = Rearrange('b t (c p1 p2) -> b t c p1 p2', p1=int(small_dim ** 0.5), p2=int(small_dim ** 0.5))
        self.temp_schedule = np.concatenate((
            np.linspace(warmup_temp,
                        temp, warmup_temp_epochs),
            np.ones(args.epochs - warmup_temp_epochs) * temp
        ))

        if init:
            self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    # ----------------------------------
    # frames simple function
    # ----------------------------------
    def f(self, n, sn):
        SL = lambda n, sn: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
                                                                                   range(int(n * i / sn),
                                                                                         max(int(n * i / sn) + 1,
                                                                                             int(n * (
                                                                                                     i + 1) / sn))))
                           for i in range(sn)]
        return SL(n, sn)

    def forward(self, img):  # img size: [2, 64, 1024]
        # ----------------------------------
        # Recoupling:
        # ----------------------------------
        if self._args.recoupling:
            img, spatial_weights, visweight = self.rcm(img.permute(0, 2, 1))
            img = img.permute(0, 2, 1)
        else:
            visweight = img

        # ----------------------------------
        sl_low = self.f(img.size(1), self.low_frames)
        xs = img[:, sl_low, :]
        b, n, _ = xs.shape

        cls_token_small = repeat(self.cls_token_small, '() n d -> b n d', b=b)
        xs = torch.cat((cls_token_small, xs), dim=1)
        xs += self.pos_embedding_small[:, :(n + 1)]
        xs = self.dropout_small(xs)

        # ----------------------------------
        sl_media = self.f(img.size(1), self.media_frames)
        xm = img[:, sl_media, :]
        b, n, _ = xm.shape

        cls_token_media = repeat(self.cls_token_media, '() n d -> b n d', b=b)
        xm = torch.cat((cls_token_media, xm), dim=1)
        xm += self.pos_embedding_media[:, :(n + 1)]
        xm = self.dropout_media(xm)

        # ----------------------------------
        sl_high = self.f(img.size(1), self.high_frames)
        xl = img[:, sl_high, :]
        b, n, _ = xl.shape

        cls_token_large = repeat(self.cls_token_large, '() n d -> b n d', b=b)
        xl = torch.cat((cls_token_large, xl), dim=1)
        xl += self.pos_embedding_large[:, :(n + 1)]
        xl = self.dropout_large(xl)

        # ----------------------------------
        # Temporal Multi-scale features learning
        # ----------------------------------
        Local_flag = True
        for multi_scale_transformer in self.multi_scale_transformers:
            xs, xm, xl = multi_scale_transformer(xs, xm, xl, Local_flag)
            Local_flag = False

        xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]
        xm = xm.mean(dim=1) if self.pool == 'mean' else xm[:, 0]
        xl = xl.mean(dim=1) if self.pool == 'mean' else xl[:, 0]

        if self._args.recoupling:
            T = self._args.temper
            distillation_loss = F.kl_div(F.log_softmax(spatial_weights.squeeze() / T, dim=-1),
                                         F.softmax(((xs + xm + xl) / 3.).detach() / T, dim=-1),
                                         reduction='sum')
        else:
            distillation_loss = torch.zeros(1).cuda()

        if self._args.Network != 'FusionNet':
            if self._args.sharpness:
                temp = self.temp_schedule[self._args.epoch]
                xs = self.mlp_head_small(xs) / temp
                xm = self.mlp_head_media(xm) / temp
                xl = self.mlp_head_large(xl) / temp
            else:
                xs = self.mlp_head_small(xs)
                xm = self.mlp_head_media(xm)
                xl = self.mlp_head_large(xl)

        if self.branch_merge == 'sum':
            x = xs + xm + xl
        elif self.branch_merge == 'pool':
            x = self.max_pool(torch.cat((xs.unsqueeze(2), xm.unsqueeze(2), xl.unsqueeze(2)), dim=-1)).squeeze()

        # ---------------------------------
        # Get score from multi-branch Trans for visualization
        # ---------------------------------
        scores_small = self.multi_scale_transformers[2].transformer_enc_small.layers[-1][0].fn.scores
        scores_media = self.multi_scale_transformers[2].transformer_enc_media.layers[-1][0].fn.scores
        scores_large = self.multi_scale_transformers[2].transformer_enc_large.layers[-1][0].fn.scores

        # resize attn
        attn_media = scores_media.detach().clone()
        attn_media.resize_(*scores_small.size())

        attn_large = scores_large.detach().clone()
        attn_large.resize_(*scores_small.size())

        att_small = scores_small.detach().clone()

        scores = torch.cat((att_small, attn_media, attn_large), dim=1)  # [2, 24, 17, 17]
        att_map = torch.zeros(scores.size(0), scores.size(1), scores.size(1), dtype=torch.float)
        for b in range(scores.size(0)):
            for i, s1 in enumerate(scores[b]):
                for j, s2 in enumerate(scores[b]):
                    cosin_simil = torch.cosine_similarity(s1.view(1, -1), s2.view(1, -1))
                    att_map[b][i][j] = cosin_simil

        # --------------------------------
        # Measure cosine similarity of xs and xl
        # --------------------------------
        cosin_similar_xs_xm = torch.cosine_similarity(xs[0], xm[0], dim=-1)
        cosin_similar_xs_xl = torch.cosine_similarity(xs[0], xl[0], dim=-1)
        cosin_similar_xm_xl = torch.cosine_similarity(xm[0], xl[0], dim=-1)
        cosin_similar_sum = cosin_similar_xs_xm + cosin_similar_xs_xl + cosin_similar_xm_xl

        return (x, xs, xm, xl), distillation_loss, (att_map, cosin_similar_sum.cpu(),
                                                    (scores_small[0], scores_media[0], scores_large[0]), visweight[0])