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
np.random.seed(123)

class JesterData(Datasets):
    def __init__(self, args, ground_truth, modality, phase='train'):
        super(JesterData, self).__init__(args, ground_truth, modality, phase)

    def LoadKeypoints(self):
        if self.phase == 'train':
            kpt_file = os.path.join(self.dataset_root, self.args.splits, 'train_kp.data')
        else:
            kpt_file = os.path.join(self.dataset_root, self.args.splits, 'valid_kp.data')
        with open(kpt_file, 'r') as f:
            kpt_data = [(lambda arr: (os.path.join(self.dataset_root, self.typ, self.phase, arr[0]), list(map(lambda x: int(float(x)), arr[1:]))))(l[:-1].split()) for l in f.readlines()]
        kpt_data = dict(kpt_data)

        for k, v in kpt_data.items():
            pose = v[:18*2]
            r_hand = v[18*2: 18*2+21*2]
            l_hand = v[18*2+21*2: 18*2+21*2+21*2]
            kpt_data[k] = {'people': [{'pose_keypoints_2d': pose, 'hand_right_keypoints_2d': r_hand, 'hand_left_keypoints_2d': l_hand}]}

        logging.info('Load Keypoints files Done, Total: {}'.format(len(kpt_data)))
        return kpt_data
    def get_path(self, imgs_path, a):
        return os.path.join(imgs_path, "%05d.jpg" % int(a + 1))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])
        self.data_path = os.path.join(self.dataset_root, self.inputs[index][0])
        # self.clip = self.image_propose(self.data_path, sl)
        self.clip, skgmaparr = self.image_propose(self.data_path, sl)

        return self.clip.permute(0, 3, 1, 2), skgmaparr, self.inputs[index][2], self.data_path

    def __len__(self):
        return len(self.inputs)
