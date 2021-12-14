'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
import math
import torch.nn.functional as F
from .utils import cosine_scheduler
import matplotlib.pyplot as plt
import numpy as np


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

def build_optim(args, model):
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate
        )
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate
        )
    return optimizer
#
def build_scheduler(args, optimizer):
    if args.scheduler['name'] == 'cosin':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs-args.scheduler['warm_up_epochs']), eta_min=args.learning_rate_min)
    elif args.scheduler['name'] == 'ReduceLR':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=args.scheduler['patience'], verbose=True,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=3, min_lr=0.00001,
                                                               eps=1e-08)
    else:
        raise NameError('build scheduler error!')

    if args.scheduler['warm_up_epochs'] > 0:
        warmup_schedule = lambda epoch: np.linspace(1e-8, args.learning_rate, args.scheduler['warm_up_epochs'])[epoch]
        return (scheduler, warmup_schedule)
    return (scheduler,)

def build_loss(args):
    loss_Function=dict(
    CE_smooth = LabelSmoothingCrossEntropy(),
    CE = torch.nn.CrossEntropyLoss(),
    MSE = torch.nn.MSELoss(),
    BCE = torch.nn.BCELoss(),
    )
    if args.loss['name'] == 'CE' and args.loss['labelsmooth']:
        return loss_Function['CE_smooth']
    return loss_Function[args.loss['name']]
