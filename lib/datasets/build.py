'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from .distributed_sampler import DistributedSampler
from .IsoGD import IsoGDData
from .NvGesture import NvData
from .THU_READ import THUREAD
from .Jester import JesterData
from .NTU import NTUData
import logging

def build_dataset(args, phase):
    modality = dict(
        M='rgb',
        K='depth',
        F='Flow'
    )
    assert args.type in modality, 'Error in modality!'
    Datasets_func = dict(
        NvGesture=NvData,
        IsoGD=IsoGDData,
        THUREAD=THUREAD,
        Jester=JesterData,
        NTU=NTUData
    )
    assert args.dataset in Datasets_func, 'Error in dataset Function!'
    if args.local_rank == 0:
        logging.info('Dataset:{}, Modality:{}'.format(args.dataset, modality[args.type]))

    if args.dataset in ['THUREAD'] and args.type == 'K':
        splits = args.splits + '/depth_{}_lst.txt'.format(phase)
    else:
        splits = args.splits + '/{}.txt'.format(phase)
    dataset = Datasets_func[args.dataset](args, splits, modality[args.type], phase=phase)
    if args.dist:
        data_sampler = DistributedSampler(dataset)
    else:
        data_sampler = None

    if phase == 'train':
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=(data_sampler is None),
                                                  sampler=data_sampler, pin_memory=True), data_sampler
    else:
        # if args.eval_only and args.nprocs == 1:
        #     args.test_batch_size = 8
        return torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                                  shuffle=False,
                                                  sampler=data_sampler, pin_memory=True, drop_last=False if args.eval_only else True), data_sampler