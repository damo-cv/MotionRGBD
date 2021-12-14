'''
This file is modified from:
https://github.com/deepmind/kinetics-i3d/i3d.py
'''

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import os, math
import sys
from .DTN import DTNNet
from .FRP import FRP_Module
from .utils import *

import os, math
import sys
sys.path.append('../../')
from collections import OrderedDict
from utils import load_pretrained_checkpoint
import logging


class DSNNet(nn.Module):
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',

        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c'
    )

    def __init__(self, args, num_classes=400, spatial_squeeze=True, name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 pretrained: str = False):

        super(DSNNet, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self.logits = None
        self.args = args

        self.end_points = {}

        '''
        Low Level Features Extraction
        '''
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[1, 7, 7],
                                            stride=(1, 2, 2), padding=(0, 3, 3), name=name + end_point)

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[1, 3, 3],
                                            padding=(0, 1, 1),
                                            name=name + end_point)

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)

        '''
        Spatial Multi-scale Features Learning
        '''
        end_point = 'Mixed_3b'
        self.end_points[end_point] = SpatialInceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)

        end_point = 'Mixed_3c'
        self.end_points[end_point] = SpatialInceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)

        end_point = 'Mixed_4b'
        self.end_points[end_point] = SpatialInceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)

        end_point = 'Mixed_4c'
        self.end_points[end_point] = SpatialInceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 2, 2], stride=(1, 2, 2),
                                                          padding=0)

        end_point = 'Mixed_5b'
        self.end_points[end_point] = SpatialInceptionModule(160 + 224 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)

        end_point = 'Mixed_5c'
        self.end_points[end_point] = SpatialInceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)

        self.LinearMap = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),

        )

        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.build()
        self.dtn = DTNNet(args, num_classes=self._num_classes)
        self.rrange = Rearrange('b c t h w -> b t c h w')

        if args.frp:
            self.frp_module = FRP_Module(w=args.w, inplanes=64)

        if pretrained:
            load_pretrained_checkpoint(self, pretrained)

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x=None, garr=None, endpoint=None):
        if endpoint == 'spatial':
            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    if end_point in ['Mixed_3b']:
                        x = self._modules[end_point](x)
                        if self.args.frp:
                            x = self.frp_module(x, garr) + x
                    elif end_point in ['Mixed_4b']:
                        x = self._modules[end_point](x)
                        if self.args.frp:
                            x = self.frp_module(x, garr) + x
                        f = x
                    elif end_point in ['Mixed_5b']:
                        x = self._modules[end_point](x)
                        if self.args.frp:
                            x = self.frp_module(x, garr) + x
                    else:
                        x = self._modules[end_point](x)

            x = self.avg_pool(x).view(x.size(0), x.size(1), -1).permute(0, 2, 1)
            x = self.LinearMap(x)
            return x
        else:
            logits, distillation_loss, (att_map, cosin_similar, MHAS, visweight) = self.dtn(x)
            return logits, distillation_loss, (att_map, cosin_similar, MHAS, visweight)
