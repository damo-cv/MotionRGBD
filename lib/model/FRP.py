'''
This file is modified from:
https://github.com/zhoubenjia/RAAR3DNet/blob/master/Network_Train/lib/model/RAAR3DNet.py
'''

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
from torchvision.utils import save_image, make_grid

def tensor_split(t):
    arr = torch.split(t, 1, dim=2)
    arr = [x.squeeze(2) for x in arr]
    return arr

def tensor_merge(arr):
    arr = [x.unsqueeze(1) for x in arr]
    t = torch.cat(arr, dim=1)
    return t.permute(0, 2, 1, 3, 4)

class FRP_Module(nn.Module):
    def __init__(self, w, inplanes):
        super(FRP_Module, self).__init__()
        self._w = w
        self.rpconv1d = nn.Conv1d(2, 1, 1, bias=False)  # Rank Pooling Conv1d, Kernel Size 2x1x1
        self.rpconv1d.weight.data = torch.FloatTensor([[[1.0], [0.0]]])
        # self.bnrp = nn.BatchNorm3d(inplanes)  # BatchNorm Rank Pooling
        # self.relu = nn.ReLU(inplace=True)
        self.hapooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, datt=None):
        inp = x
        if self._w < 1:
            return x
        def run_layer_on_arr(arr, l):
            return [l(x) for x in arr]
        def oneconv(a, b):
            s = a.size()
            c = torch.cat([a.contiguous().view(s[0], -1, 1), b.contiguous().view(s[0], -1, 1)], dim=2)
            c = self.rpconv1d(c.permute(0, 2, 1)).permute(0, 2, 1)
            return c.view(s)
        if datt is not None:
            tarr = tensor_split(x)
            garr = tensor_split(datt)
            while tarr[0].size()[3] < garr[0].size()[3]:  # keep feature map and heatmap the same size
                garr = run_layer_on_arr(garr, self.hapooling)

            attarr = [a * (b + torch.ones(a.size()).cuda()) for a, b in zip(tarr, garr)]
            datt = [oneconv(a, b) for a, b in zip(tarr, attarr)]
            return tensor_merge(datt)

        def tensor_arr_rp(arr):
            l = len(arr)
            def tensor_rankpooling(video_arr):
                def get_w(N):
                    return [float(i) * 2 - N - 1 for i in range(1, N + 1)]

                # re = torch.zeros(video_arr[0].size(0), 1, video_arr[0].size(2), video_arr[0].size(3)).cuda()
                re = torch.zeros(video_arr[0].size()).cuda()
                for a, b in zip(video_arr, get_w(len(video_arr))):
                    # a = transforms.Grayscale(1)(a)
                    re += a * b
                re = F.gelu(re)
                re -= torch.min(re)
                re = re / torch.max(re) if torch.max(re) != 0 else re / (torch.max(re) + 0.00001)
                return transforms.Grayscale(1)(re)

            return [tensor_rankpooling(arr[i:i + self._w]) for i in range(l)]

        arrrp = tensor_arr_rp(tensor_split(x))

        b, c, t, h, w = tensor_merge(arrrp).shape
        mask = torch.zeros(b, c, self._w-1,  h, w, device=tensor_merge(arrrp).device)
        garrs = torch.cat((mask,  tensor_merge(arrrp)), dim=2)
        return garrs

if __name__ == '__main__':
    model = SATT_Module().cuda()
    inp = torch.randn(2, 3, 64, 224, 224).cuda()
    out = model(inp)
    print(out.shape)
