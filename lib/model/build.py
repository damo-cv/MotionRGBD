'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

from .DSN import DSNNet
from .fusion_Net import CrossFusionNet

import logging

def build_model(args):
    num_classes = dict(
        IsoGD=249,
        NvGesture=25,
        Jester=27,
        THUREAD=40,
        NTU=60
    )
    func_dict = dict(
        I3DWTrans=DSNNet,
        FusionNet=CrossFusionNet
    )
    assert args.dataset in num_classes, 'Error in load dataset !'
    assert args.Network in func_dict, 'Error in Network function !'
    args.num_classes = num_classes[args.dataset]
    if args.local_rank == 0:
        logging.info('Model:{}, Total Categories:{}'.format(args.Network, args.num_classes))
    return func_dict[args.Network](args, num_classes=args.num_classes, pretrained=args.pretrained)
