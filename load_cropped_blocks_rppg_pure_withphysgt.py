import os
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import torch
import pickle

class load_cropped_pure(Dataset):
    def __init__(self,rppg_net,data,shuffle=False, Training=True, transform=None):
        self.train = Training
        self.data = data
        self.transform = transform
        random.seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        if shuffle:
            random.shuffle(self.data)
        if rppg_net == "physformer":
            with open('Physformer_pure_signals_train.pkl', 'rb') as f:
                list_names = pickle.load(f)
                signals_train = pickle.load(f)
            with open('Physformer_pure_signals_validate.pkl', 'rb') as f:
                list_names = pickle.load(f)
                signals_validate = pickle.load(f)
        if rppg_net == "physnet":
            with open('Physnet_pure_signals_train.pkl', 'rb') as f:
                list_names = pickle.load(f)
                signals_train = pickle.load(f)
            with open('Physnet_pure_signals_validate.pkl', 'rb') as f:
                list_names = pickle.load(f)
                signals_validate = pickle.load(f)
        if rppg_net == "chrom"  or rppg_net == "none":
            with open('Chrom_pure_signals_train.pkl', 'rb') as f:
                list_names = pickle.load(f)
                signals_train = pickle.load(f)
            with open('Chrom_pure_signals_validate.pkl', 'rb') as f:
                list_names = pickle.load(f)
                signals_validate = pickle.load(f)

        self.signals = signals_train | signals_validate

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        T = 64
        offset = 60
        sample = self.data[idx]
        dir = sample[0]
        id = int(sample[1])-1 #so it starts from 0
        pics = np.load(dir+"/"+sample[2])
        frame0 = int(sample[3][:-4])
        gender = int(sample[4])
        eth = int(sample[5])
        imgs = []

        video_tensor = torch.FloatTensor(pics/255)
        mask = np.load(dir[:-6]+"mask/"+sample[2])
        mask_tensor = torch.FloatTensor(mask)
        #bvp = np.load(dir[:-6]+"bvp.npy")
        lnd = np.load(dir[:-6]+"lnd.npy")
        #bvp = bvp[frame0-offset:frame0-offset+T]

        name = dir.split('/')[-2]+"__"+str(frame0).zfill(4)
        bvp = self.signals[name]
        lnd = lnd[frame0-offset:frame0-offset+T]
        bvp = (bvp - np.min(bvp))/(np.max(bvp)-np.min(bvp))
        return video_tensor, bvp, id, gender, eth, dir, frame0, lnd,mask_tensor
