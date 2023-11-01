import os
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
import math

class Acc(object):
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.error = 0
        self.mae = 0
        self.std = 0
        self.rmse = 0
        self.sum = 0
        self.count = 0
        self.sqr_sum = 0

    def update(self, error, n=1):
        self.error = error
        self.sum += np.sum(np.abs(error)).item()
        self.sqr_sum += np.sum(np.square(error)).item()
        self.count += n
        self.mae = self.sum / self.count
        self.rmse = math.sqrt(self.sqr_sum / self.count)
        self.std = math.sqrt(self.sqr_sum / self.count-(self.sum / self.count)**2)

class AverageMeter(object):
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
