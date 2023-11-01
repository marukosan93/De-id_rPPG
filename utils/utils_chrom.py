import torch
import numpy as np
import matplotlib.pyplot as plt

loss_global_train = []
loss_global_valid = []
acc_global_train = []
acc_global_valid = []


def running_mean_gpu(x, N):
    out = torch.zeros_like(x)
    dim_len = x.size()[1]
    for i in range(0,dim_len):
        if N%2 == 0:
            a, b = i - (N-1)//2, i + (N-1)//2 + 2
        else:
            a, b = i - (N-1)//2, i + (N-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[:,i] = torch.mean(x[:,a:b],axis=1)
    return out


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
    bvp = running_mean_gpu(bvp,5)
    return bvp


def raw_rppg(masked):
    num = torch.sum(masked,(3,4))
    denom = torch.count_nonzero(masked,(3,4))
    result = num/denom
    return result

