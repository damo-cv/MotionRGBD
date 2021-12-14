'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from .base import Datasets
from torchvision import transforms, set_image_backend
import random, os
from PIL import Image
import numpy as np

class NTUData(Datasets):
    def __init__(self, args, ground_truth, modality, phase='train'):
        super(NTUData, self).__init__(args, ground_truth, modality, phase)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])

        if self.typ == 'rgb':
            self.data_path = os.path.join(self.dataset_root, 'Images', self.inputs[index][0])

        if self.typ == 'depth':
            self.data_path = os.path.join(self.dataset_root, 'nturgb+d_depth_masked', self.inputs[index][0][:-4])

        self.clip, skgmaparr = self.image_propose(self.data_path, sl)

        if self.args.Network == 'FusionNet':
            assert self.typ == 'rgb'
            self.data_path = os.path.join(self.dataset_root, 'nturgb+d_depth_masked', self.inputs[index][0][:-4])
            self.clip1, skgmaparr1 = self.image_propose(self.data_path, sl)
            return (self.clip.permute(0, 3, 1, 2), self.clip1.permute(0, 3, 1, 2)), (skgmaparr, skgmaparr1), \
                   self.inputs[index][2], self.data_path

        return self.clip.permute(0, 3, 1, 2), skgmaparr, self.inputs[index][2], self.inputs[index][0]

    def get_path(self, imgs_path, a):

        if self.typ == 'rgb':
            return os.path.join(imgs_path, "%06d.jpg" % int(a + 1))
        else:
            return os.path.join(imgs_path, "MDepth-%08d.png" % int(a + 1))

    def __len__(self):
        return len(self.inputs)
