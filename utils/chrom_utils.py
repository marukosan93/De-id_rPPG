import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
from torchvision import models
import imageio.v2 as imageio
import pickle
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as op
import time
import random
from scipy.fft import rfft
from scipy import signal
import math
import os
import argparse
import glob
import torch.backends.cudnn as cudnn
from scipy.signal import butter, filtfilt, resample, sosfiltfilt, welch, detrend
from scipy.fft import fft
from utils.signals_stuff import butter_bandpass, norm, hr_fft,NegPearson
import sys
from PIL import Image, ImageDraw
from torch.nn.functional import normalize

def shrinkroi(landsss,scala):
    shift = np.min(landsss,axis=0)+(np.max(landsss,axis=0)-np.min(landsss,axis=0))/2
    landsss = landsss - shift
    landsss = landsss * scala
    landsss = landsss + shift
    return landsss

def poly2mask(polyarray,m,n):
    img = Image.new('L', (n, m), 0)
    ImageDraw.Draw(img).polygon(polyarray.flatten().tolist(), outline=1, fill=1)
    mask = np.array(img)
    return mask


def create_mask(lnd,bctwh):
    final_mask = torch.zeros(bctwh)
    batch,channel,temp,n,m = bctwh

    ROI_cheek_left1 = np.array([0,1,2,31,41,0])
    ROI_cheek_left2 = np.array([2,3,4,5,48,31,2])
    ROI_cheek_right1 = np.array([16,15,14,35,46,16])
    ROI_cheek_right2 = np.array([14,13,12,11,54,35,14])
    ROI_mouth = [5,6,7,8,9,10,11,54,55,56,57,58,59,48,5]
    ROI_forehead = [17,18,19,20,21,22,23,24,25,26]

    for b in range(0,batch):
        for t in range(0,temp):
            lmks = lnd[b,t,:,:]
            forehead = lmks[ROI_forehead]
            left_eye = np.mean(lmks[36:42],axis=0)
            right_eye = np.mean(lmks[42:48],axis=0)
            eye_distance = np.linalg.norm(left_eye-right_eye)

            tmp = (np.mean(lmks[17:22],axis=0)+ np.mean(lmks[22:27],axis=0))/2 - (left_eye + right_eye)/2;
            tmp = (eye_distance/np.linalg.norm(tmp))*0.6*tmp;

            scala = 0.9
            lmks[ROI_cheek_left1] = shrinkroi(lmks[ROI_cheek_left1],scala)
            lmks[ROI_cheek_left2] = shrinkroi(lmks[ROI_cheek_left2],scala)
            lmks[ROI_cheek_right1] = shrinkroi(lmks[ROI_cheek_right1],scala)
            lmks[ROI_cheek_right2] = shrinkroi(lmks[ROI_cheek_right2],scala)
            lmks[ROI_mouth] = shrinkroi(lmks[ROI_mouth],scala)
            new_ROI_forehead = shrinkroi(ROI_forehead,scala)

            new_ROI_forehead=(np.vstack((forehead,forehead[-1].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)))).round(0).astype(int)
            mask_ROI_cheek_left1 = poly2mask(lmks[ROI_cheek_left1],m,n);
            mask_ROI_cheek_left2 = poly2mask(lmks[ROI_cheek_left2],m,n);
            mask_ROI_cheek_right1 = poly2mask(lmks[ROI_cheek_right1],m,n);
            mask_ROI_cheek_right2 = poly2mask(lmks[ROI_cheek_right2],m,n);
            mask_ROI_mouth  = poly2mask(lmks[ROI_mouth],m,n);
            mask_ROI_forehead = poly2mask(new_ROI_forehead,m,n);

            mask = np.clip(mask_ROI_cheek_left1+mask_ROI_cheek_left2+mask_ROI_cheek_right1+mask_ROI_cheek_right2+mask_ROI_forehead+mask_ROI_mouth,0,1)
            mask = np.stack([mask]*3,axis=0)
            final_mask[b,:,t,:,:] = torch.Tensor(mask)
    return final_mask

def gpu_CHROM(X):
    Xcomp = 3*X[:,0,:]- 2*X[:,1,:]
    Ycomp = (1.5*X[:,0,:])+X[:,1,:]-(1.5*X[:,2,:])
    sX = torch.std(Xcomp,axis=-1)
    sY = torch.std(Ycomp,axis=-1)
    alpha = (sX/sY)
    alpha = torch.stack([alpha]*64,dim=1)
    bvp = Xcomp-alpha*Ycomp
    #bvp = butter_bandpass(bvp,0.7,3,30,order=3)
    minimo = torch.min(bvp,dim=1)[0]
    massimo = torch.max(bvp,dim=1)[0]
    minimo = torch.stack([minimo]*64,dim=1)
    massimo = torch.stack([massimo]*64,dim=1)
    bvp = (bvp - minimo)/(massimo-minimo)
    return bvp

def raw_rppg(masked):
    num = torch.sum(masked,(3,4))
    denom = torch.count_nonzero(masked,(3,4))
    result = num/denom
    return result

def shrinkroi(landsss,scala):
    shift = np.min(landsss,axis=0)+(np.max(landsss,axis=0)-np.min(landsss,axis=0))/2
    landsss = landsss - shift
    landsss = landsss * scala
    landsss = landsss + shift
    return landsss

def poly2mask(polyarray,m,n):
    img = Image.new('L', (n, m), 0)
    ImageDraw.Draw(img).polygon(polyarray.flatten().tolist(), outline=1, fill=1)
    mask = np.array(img)
    return mask


def create_mask(lnd,bctwh):
    lnd = lnd.detach().cpu().numpy()
    final_mask = torch.zeros(bctwh)
    batch,channel,temp,n,m = bctwh

    ROI_cheek_left1 = np.array([0,1,2,31,41,0])
    ROI_cheek_left2 = np.array([2,3,4,5,48,31,2])
    ROI_cheek_right1 = np.array([16,15,14,35,46,16])
    ROI_cheek_right2 = np.array([14,13,12,11,54,35,14])
    ROI_mouth = [5,6,7,8,9,10,11,54,55,56,57,58,59,48,5]
    ROI_forehead = [17,18,19,20,21,22,23,24,25,26]

    scala = 0.9
    for b in range(0,batch):
        for t in range(0,temp):
            lmks = lnd[b,t,:,:]
            forehead = lmks[ROI_forehead]
            left_eye = np.mean(lmks[36:42],axis=0)
            right_eye = np.mean(lmks[42:48],axis=0)
            eye_distance = np.linalg.norm(left_eye-right_eye)

            tmp = (np.mean(lmks[17:22],axis=0)+ np.mean(lmks[22:27],axis=0))/2 - (left_eye + right_eye)/2;
            tmp = (eye_distance/np.linalg.norm(tmp))*0.6*tmp;

            lmks[ROI_cheek_left1] = shrinkroi(lmks[ROI_cheek_left1],scala)
            lmks[ROI_cheek_left2] = shrinkroi(lmks[ROI_cheek_left2],scala)
            lmks[ROI_cheek_right1] = shrinkroi(lmks[ROI_cheek_right1],scala)
            lmks[ROI_cheek_right2] = shrinkroi(lmks[ROI_cheek_right2],scala)
            lmks[ROI_mouth] = shrinkroi(lmks[ROI_mouth],scala)
            new_ROI_forehead = shrinkroi(ROI_forehead,scala)

            new_ROI_forehead=(np.vstack((forehead,forehead[-1].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)))).round(0).astype(int)
            mask_ROI_cheek_left1 = poly2mask(lmks[ROI_cheek_left1],m,n);
            mask_ROI_cheek_left2 = poly2mask(lmks[ROI_cheek_left2],m,n);
            mask_ROI_cheek_right1 = poly2mask(lmks[ROI_cheek_right1],m,n);
            mask_ROI_cheek_right2 = poly2mask(lmks[ROI_cheek_right2],m,n);
            mask_ROI_mouth  = poly2mask(lmks[ROI_mouth],m,n);
            mask_ROI_forehead = poly2mask(new_ROI_forehead,m,n);

            mask = np.clip(mask_ROI_cheek_left1+mask_ROI_cheek_left2+mask_ROI_cheek_right1+mask_ROI_cheek_right2+mask_ROI_forehead+mask_ROI_mouth,0,1)
            mask = np.stack([mask]*3,axis=0)
            final_mask[b,:,t,:,:] = torch.Tensor(mask)
    return final_mask
