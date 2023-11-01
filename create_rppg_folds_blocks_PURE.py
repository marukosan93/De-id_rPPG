import os
import time
import random
import numpy as np
import torch
import more_itertools as mit
import pickle
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch
import matplotlib.pyplot as plt
from utils.signals_stuff import hr_fft

#input_dir = "./PURE_cropped_faces/"
input_dir = "./PURE_cropped_tight/"
video_list = os.listdir(input_dir) #maybe use list_dirs custom function if things get weird

#video_list = video_list[:10]
train_samples_list = []
valid_samples_list = []
T = 64
offset = 60

labels_dict = {}

with open('./folds/pure_labels_gen_eth.txt') as f:
    lines = f.readlines()
for l in lines[1:]:
    tmp = l[:-1].split(" ")
    labels_dict[tmp[0]] = tmp[1]+tmp[2]

for v in video_list:
    video_samples_list = []
    img_folder_path = os.path.join(input_dir,v,"blocks")
    n = len(os.listdir(img_folder_path))
    images = []
    splittino = v.split("-")
    id = splittino[0]
    vidnum = splittino[1]
    label = labels_dict[id]

    bvp = np.load(img_folder_path[:-6]+"bvp.npy")

    k = int(len(bvp)/T)
    for i in range(0,k):
        vid = i*T+offset
        bvp_part = bvp[vid-offset:vid-offset+T]
        #hr_ = hr_fft(bvp_part)
        #if hr_ > 40 and hr_ < 180:
            #print(hr_)

        video_samples_list.append((img_folder_path,id,str(i).zfill(3)+".npy",str(vid).zfill(4)+".png",label[0],label[1]))
    if vidnum == "01":
        valid_samples_list += video_samples_list
    else:
        train_samples_list += video_samples_list

print(len(train_samples_list))
print(len(valid_samples_list))
file = open("./folds/pure_folds_rppg.pkl",'wb')
pickle.dump(train_samples_list, file)
pickle.dump(valid_samples_list, file)
