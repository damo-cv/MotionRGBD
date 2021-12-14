'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

import numpy as np

import os
import sys
from collections import OrderedDict

sys.path.append(['../../', '../'])
from utils import load_pretrained_checkpoint, load_checkpoint
import logging
from .DSN_Fusion import DSNNet

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


class Encoder(nn.Module):
    def __init__(self, C_in, C_out, dilation=2):
        super(Encoder, self).__init__()
        self.enconv = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=False),

            nn.Conv2d(C_in, C_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in // 2),
            nn.ReLU(inplace=False),

            nn.Conv2d(C_in // 2, C_in // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in // 4),
            nn.ReLU(inplace=False),

            nn.Conv2d(C_in // 4, C_out, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x1, x2):
        b, c = x1.shape
        x = torch.cat((x1, x2), dim=1).view(b, -1, 1, 1)
        x = self.enconv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, C_in, C_out, dilation=2):
        super(Decoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.Conv2d(C_in, C_out // 4, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out // 4),
            nn.ReLU(),

            nn.Conv2d(C_out // 4, C_out // 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x


class FusionModule(nn.Module):
    def __init__(self, channel_in=1024, channel_out=256, num_classes=60):
        super(FusionModule, self).__init__()
        self.encoder = Encoder(channel_in, channel_out)
        self.decoder = Decoder(channel_out, channel_in)
        self.efc = nn.Conv2d(channel_out, num_classes, kernel_size=1, padding=0, bias=False)

    def forward(self, r, d):
        en_x = self.encoder(r, d)  # [4, 256, 1, 1]
        de_x = self.decoder(en_x)
        en_x = self.efc(en_x)
        return en_x.squeeze(), de_x

class CrossFusionNet(nn.Module):
    def __init__(self, args, num_classes, pretrained, spatial_interact=True, temporal_interact=True):
        super(CrossFusionNet, self).__init__()
        self._MES = torch.nn.MSELoss()
        self._BCE = torch.nn.BCELoss()
        self._CE = LabelSmoothingCrossEntropy()
        self.spatial_interact = spatial_interact
        self.temporal_interact = temporal_interact

        self.fusion_model = FusionModule(channel_out=256, num_classes=num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False)
        self.dropout = nn.Dropout(0.5)

        assert args.rgb_checkpoint and args.depth_checkpoint
        self.Modalit_rgb = DSNNet(args, num_classes=num_classes,
                                     pretrained=args.rgb_checkpoint)

        self.Modalit_depth = DSNNet(args, num_classes=num_classes,
                                       pretrained=args.depth_checkpoint)

        if self.spatial_interact:
            self.crossFusion = nn.Sequential(
                nn.Conv2d(512 * 2, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout(0.4)
                
            )
        if self.temporal_interact:
            self.crossFusionT = nn.Sequential(
                nn.Conv2d(512 * 2, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout(0.4)
            )

        self.classifier1 = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )

        if pretrained:
            load_pretrained_checkpoint(self, pretrained)
            logging.info("Load Pre-trained model state_dict Done !")

    def forward(self, inputs, garrs, target):
        rgb, depth = inputs
        rgb_garr, depth_garr = garrs

        spatial_M = self.Modalit_rgb(rgb, rgb_garr, endpoint='spatial')
        spatial_K = self.Modalit_depth(depth, depth_garr, endpoint='spatial')

        if self.spatial_interact:
            b, t, c = spatial_M.shape
            spatial_fusion_features = self.crossFusion(F.normalize(torch.cat((spatial_M, spatial_K), dim=-1), p = 2, dim=-1).view(b, c*2, t, 1)).squeeze()

        (temporal_M, M_xs, M_xm, M_xl), distillationM, _ = self.Modalit_rgb(x=spatial_M + spatial_fusion_features.view(spatial_M.shape) if self.spatial_interact else spatial_M,
                                                                         endpoint='temporal')  # size[4, 512]
        (temporal_K, K_xs, K_xm, K_xl), distillationK, _ = self.Modalit_depth(x=spatial_K + spatial_fusion_features.view(spatial_M.shape) if self.spatial_interact else spatial_K,
                                                                           endpoint='temporal')
        logit_r = self.classifier1(temporal_M)
        logit_d = self.classifier2(temporal_K)

        if self.temporal_interact:
            b, c = temporal_M.shape
            temporal_fusion_features = self.crossFusionT(F.normalize(torch.cat((temporal_M, temporal_K), dim=-1), p = 2, dim=-1).view(b, c*2, 1, 1)).squeeze()
            temporal_M, temporal_K = temporal_M+temporal_fusion_features, temporal_K+temporal_fusion_features

        en_x, de_x = self.fusion_model(temporal_M, temporal_K)
        b, c = temporal_M.shape
        bce_r = torch.sigmoid(self.fc(self.dropout(temporal_M).view(b, c, 1, 1))).view(b, -1)
        bce_d = torch.sigmoid(self.fc(self.dropout(temporal_K).view(b, c, 1, 1))).view(b, -1)

        BCE_loss = self._BCE(bce_r, torch.ones(bce_r.size(0), 1).cuda()) + self._BCE(bce_d, torch.zeros(bce_d.size(0),
                                                                                                        1).cuda())
        MSE_loss = self._MES(de_x.view(b, c), temporal_M) + self._MES(de_x.view(b, c), temporal_K)
        CE_loss = self._CE(en_x, target) + self._CE(logit_r, target) + self._CE(logit_d, target)
        distillation = distillationM + distillationK

        return (en_x, logit_r, logit_d), (CE_loss, BCE_loss, MSE_loss, distillation)
