'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from .base import Datasets
from torchvision import transforms, set_image_backend
import random, os
from PIL import Image
import numpy as np

class IsoGDData(Datasets):
    def __init__(self, args, ground_truth, modality, phase='train'):
        super(IsoGDData, self).__init__(args, ground_truth, modality, phase)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])
        self.data_path = os.path.join(self.dataset_root, self.typ, self.inputs[index][0])
        if self.typ == 'depth':
            self.data_path = self.data_path.replace('M_', 'K_')

        if self.args.Network == 'FusionNet':
            assert self.typ == 'rgb'
            self.data_path1 = self.data_path.replace('rgb', 'depth')
            self.data_path1 = self.data_path1.replace('M', 'K')

            self.clip, skgmaparr = self.image_propose(self.data_path, sl)
            self.clip1, skgmaparr1 = self.image_propose(self.data_path1, sl)
            return (self.clip.permute(0, 3, 1, 2), skgmaparr), (self.clip1.permute(0, 3, 1, 2), skgmaparr1), self.inputs[index][2], self.inputs[index][0]

        else:
            self.clip, skgmaparr = self.image_propose(self.data_path, sl)
            return self.clip.permute(0, 3, 1, 2), skgmaparr, self.inputs[index][2], self.inputs[index][0]

    def __len__(self):
        return len(self.inputs)
