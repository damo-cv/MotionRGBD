'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from .base import Datasets
from torchvision import transforms, set_image_backend
import random, os
from PIL import Image
import numpy as np
import logging
set_image_backend('accimage')
np.random.seed(123)

class NvData(Datasets):
    def __init__(self, args, ground_truth, modality, phase='train'):
        super(NvData, self).__init__(args, ground_truth, modality, phase)
    def transform_params(self, resize=(320, 240), crop_size=224, flip=0.5):
        if self.phase == 'train':
            left, top = random.randint(10, resize[0] - crop_size), random.randint(10, resize[1] - crop_size)
            is_flip = True if random.uniform(0, 1) < flip else False
        else:
            left, top = 32, 32
            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])
        self.data_path = os.path.join(self.dataset_root, self.typ, self.inputs[index][0])
        self.clip, skgmaparr = self.image_propose(self.data_path, sl)

        if self.args.Network == 'FusionNet':
            assert self.typ == 'rgb'
            self.data_path = self.data_path.replace('rgb', 'depth')
            self.clip1, skgmaparr1 = self.image_propose(self.data_path, sl)

            return (self.clip.permute(0, 3, 1, 2), self.clip1.permute(0, 3, 1, 2)), (skgmaparr, skgmaparr1), self.inputs[index][2], self.data_path

        return self.clip.permute(0, 3, 1, 2), skgmaparr, self.inputs[index][2], self.data_path

    def __len__(self):
        return len(self.inputs)
