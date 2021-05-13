import os
import numpy as np
from PIL import Image

import torch
from torch import nn

class AverageMeter(object):
    """Computes and stores the average and current value  计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset (self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  # ‘+=’表示两个值相加，然后返回值给符号左侧的变量
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

if __name__ == "__main__":
    a = np.array([[1,2,3,3,3,3,4,5,0,0,0],[1,2,3,2,3,1,4,5,0,0,0]])
    b = np.array([[1,2,4,3,2,3,4,5,0,2,0],[1,1,3,3,3,3,4,5,0,1,1]])
    i,u,area = intersectionAndUnion(a,b,K=6)
    print(i,u,area)
    print(i.sum()/area.sum())
    print((a==b).sum()/b.size)
