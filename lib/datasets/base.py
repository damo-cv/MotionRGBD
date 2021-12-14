'''
This file is modified from:
https://github.com/zhoubenjia/RAAR3DNet/blob/master/Network_Train/lib/datasets/base.py
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, set_image_backend
import torch.nn.functional as F

from PIL import Image
from PIL import ImageFilter, ImageOps
import os, glob
import math, random
import numpy as np
import logging
from tqdm import tqdm as tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import cv2
import json
from scipy.ndimage.filters import gaussian_filter

# import functools
import matplotlib.pyplot as plt  # For graphics
np.random.seed(123)

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class Datasets(Dataset):
    global kpt_dict
    def __init__(self, args, ground_truth, modality, phase='train'):

        def get_data_list_and_label(data_df):
            return [(lambda arr: (arr[0], int(arr[1]), int(arr[2])))(i[:-1].split(' '))
                    for i in open(data_df).readlines()]

        self.dataset_root = args.data
        self.sample_duration = args.sample_duration
        self.sample_size = args.sample_size
        self.phase = phase
        args.phase = phase
        self.typ = modality
        self.args = args
        self._w = args.w

        self.transform = transforms.Compose([Normaliztion(), transforms.ToTensor()])

        self.inputs = list(filter(lambda x: x[1] > 16, get_data_list_and_label(ground_truth)))
        self.inputs = list(self.inputs)
        if phase == 'train':
            while len(self.inputs) % (args.batch_size * args.nprocs) != 0:
                sample = random.choice(self.inputs)
                self.inputs.append(sample)
            logging.info('Training Data Size is: {}'.format(len(self.inputs)))
            frames = [n[1] for n in self.inputs]
            logging.info('Average Train Data frames are: {}, max frames: {}, min frames: {}'.format(sum(frames)//len(self.inputs), max(frames), min(frames)))
        else:
            logging.info('Validation Data Size is: {} '.format(len(self.inputs)))
            frames = [n[1] for n in self.inputs]
            logging.info('Average Train Data frames are: {}, max frames: {}, min frames: {}'.format(
                sum(frames) // len(self.inputs), max(frames), min(frames)))

    def transform_params(self, resize=(320, 240), crop_size=224, flip=0.5):
        if self.phase == 'train':
            left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
            is_flip = True if np.random.uniform(0, 1) < flip else False
        else:
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2

            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def get_path(self, imgs_path, a):
        return os.path.join(imgs_path, "%06d.jpg" % a)

    def depthProposess(self, img):
        h2, w2 = img.shape

        mask = img.copy()
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((10, 10), np.uint8))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find Max Maxtri
        Idx = []
        for i in range(len(contours)):
            Area = cv2.contourArea(contours[i])
            if Area > 500:
                Idx.append(i)
        centers = []

        for i in Idx:
            rect = cv2.minAreaRect(contours[i])
            center, (h, w), degree = rect
            centers.append(center)

        finall_center = np.int0(np.array(centers))
        c_x = min(finall_center[:, 0])
        c_y = min(finall_center[:, 1])
        center = (c_x, c_y)

        crop_x, crop_y = 320, 240
        left = center[0] - crop_x // 2 if center[0] - crop_x // 2 > 0 else 0
        top = center[1] - crop_y // 2 if center[1] - crop_y // 2 > 0 else 0
        crop_w = left + crop_x if left + crop_x < w2 else w2
        crop_h = top + crop_y if top + crop_y < h2 else h2
        rect = (left, top, crop_w, crop_h)
        image = Image.fromarray(img)
        image = image.crop(rect)
        return image

    def image_propose(self, data_path, sl):
        sample_size = self.sample_size
        resize = eval(self.args.resize)
        crop_rect, is_flip = self.transform_params(resize=resize, crop_size=self.args.crop_size, flip=self.args.flip)  # no flip
        if np.random.uniform(0, 1) < self.args.rotated and self.phase == 'train':
            r, l = eval(self.args.angle)
            rotated = np.random.randint(r, l)
        else:
            rotated = 0

        def transform(img):
            img = np.asarray(img)
            if img.shape[-1] != 3:
                img = np.uint8(255 * img)
                img = self.depthProposess(img)
                img = cv2.applyColorMap(np.asarray(img), cv2.COLORMAP_JET)
            img = self.rotate(np.asarray(img), rotated)
            img = Image.fromarray(img)
            img = img.resize(resize)
            img = img.crop(crop_rect)
            if self.args.Blur and self.args.phase == 'train':
                img = GaussianBlur()(img)
            if is_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img.resize((sample_size, sample_size)))

        def Sample_Image(imgs_path, sl):
            frams = []
            for a in sl:
                try:
                    ori_image = Image.open(self.get_path(imgs_path, a))
                except:
                    ori_image = Image.open(os.path.join(imgs_path, "MDepth-%08d.png" % int(a+1))) # For NTU fusion
                img = transform(ori_image)
                frams.append(self.transform(img).view(3, sample_size, sample_size, 1))
            skgmaparr = DynamicImage(frams, dynamic_only=False)
            return torch.cat(frams, dim=3).type(torch.FloatTensor), skgmaparr.unsqueeze(0)

        def DynamicImage(frames, dynamic_only): # frames: [[3, 224, 224, 1], ]
            def tensor_arr_rp(arr):
                l = len(arr)
                statics = []
                def tensor_rankpooling(video_arr, lamb=1.):
                    def get_w(N):
                        return [float(i) * 2 - N - 1 for i in range(1, N + 1)]

                    # re = torch.zeros(video_arr[0].size(0), 1, video_arr[0].size(2), video_arr[0].size(3)).cuda()
                    re = torch.zeros(video_arr[0].size())
                    for a, b in zip(video_arr, get_w(len(video_arr))):
                        # a = transforms.Grayscale(1)(a)
                        re += a * b
                    re = F.relu(re) * lamb
                    re -= torch.min(re)
                    re = re / torch.max(re) if torch.max(re) != 0 else re / (torch.max(re) + 0.00001)

                    re = transforms.Grayscale(1)(re.squeeze())
                    # Static Attention
                    static = torch.where(re > torch.mean(re), re, torch.full_like(re, 0))
                    static = np.asarray(static.squeeze())
                    # static = cv2.morphologyEx(static, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    static = cv2.erode(static, kernel)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    static = cv2.dilate(static, kernel)
                    static -= np.min(static)
                    static = static / np.max(static) if np.max(static) != 0 else static / (np.max(static) + 0.00001)
                    statics.append(torch.from_numpy(static).unsqueeze(0))
                    return re

                return [tensor_rankpooling(arr[i:i + self._w]) for i in range(l)], statics
            arrrp, statics = tensor_arr_rp(frames)
            arrrp = torch.cat(arrrp, dim=0) # torch.Size([64, 224, 224])
            t, h, w = arrrp.shape
            mask = torch.zeros(self._w - 1, h, w)
            garrs = torch.cat((mask, arrrp), dim=0)[:t, :]
            statics = torch.cat(statics)
            statics = torch.cat((mask, statics))[:t, :]
            if dynamic_only:
                return garrs
            return (garrs + statics) * statics
        return Sample_Image(data_path, sl)

    def get_sl(self, clip):
        sn = self.sample_duration
        if self.phase == 'train':
            f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                   range(int(n * i / sn),
                                                                                         max(int(n * i / sn) + 1,
                                                                                             int(n * (
                                                                                                     i + 1) / sn))))
                           for i in range(sn)]
        else:
            f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                    max(int(
                                                                                                        n * i / sn) + 1,
                                                                                                        int(n * (
                                                                                                                i + 1) / sn))))
                           for i in range(sn)]
        return f(int(clip))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])
        self.data_path = os.path.join(self.dataset_root, self.inputs[index][0])
        self.clip = self.image_propose(self.data_path, sl)
        return self.clip.permute(0, 3, 1, 2), self.inputs[index][2]
    def __len__(self):
        return len(self.inputs)

if __name__ == '__main__':
    import argparse
    from config import Config
    from lib import *
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='Place config Congfile!')
    parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--nprocs', type=int, default=1)

    parser.add_argument('--save_grid_image', action='store_true', help='Save samples?')
    parser.add_argument('--save_output', action='store_true', help='Save logits?')
    parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')

    parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
    parser.add_argument('--save', type=str, default='Checkpoints/', help='experiment name')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    args = parser.parse_args()
    args = Config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.dist = False
    args.eval_only = True
    args.test_batch_size = 1

    valid_queue, valid_sampler = build_dataset(args, phase='val')
    for step, (inputs, heatmap, target, _) in enumerate(valid_queue):
        print(inputs.shape)
        input()