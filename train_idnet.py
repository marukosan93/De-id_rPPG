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
from load_cropped_blocks_rppg_pure import load_cropped_pure
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as op
import time
import random
from scipy.fft import fft
from scipy import signal
import math
import os
import torch.backends.cudnn as cudnn
from utils.Average_Accuracy import AverageMeter
from utils.Average_Accuracy import Acc
import sys
from facenet_pytorch import InceptionResnetV1
from senet import senet50
from torchsummary import summary

loss_id_global_train = []
loss_id_global_valid = []
acc_id_global_train = []
acc_id_global_valid = []



def norm(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def train(train_loader, model_id, criterion_id, optimizer, epoch, name_of_run,num_classes):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_id = AverageMeter()
    acc = Acc()

    # switch to train mode
    model_id.train()

    correct = 0
    total = 0


    end = time.time()
    for i, (input,bvp,id,gender,race,dir,frame0,lnd,mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.to(device=device, dtype=torch.float)
        id = id.to(device=device, dtype=torch.long)
        label = F.one_hot(id,num_classes=num_classes).float()

        # compute output
        output = input
        t = 32
        out_id = model_id(normt(output[:,:,t,:,:]))

        pred = torch.argmax(out_id,dim=1)

        correct += ((pred == id).sum()).cpu().numpy()
        total += id.size(0)
        id_acc = correct/total

        loss_id = criterion_id(out_id,label)

        loss = loss_id
        optimizer.zero_grad()
        loss_id.backward()
        optimizer.step()

        loss = loss.float()

        # measure accuracy and record loss
        losses_id.update(loss_id.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0 or i == len(train_loader)-2:
            fig,ax = plt.subplots(BATCH_SIZE,2,figsize=(15,15))
            for r in range(0,BATCH_SIZE):
                ax[r][0].imshow(input[r,:,32,:,:].permute(1,2,0).detach().cpu().numpy())
                ax[r][1].imshow(output[r,:,32,:,:].permute(1,2,0).detach().cpu().numpy())
                ax[r][0].set_title("ID: "+str(id[r].detach().cpu().numpy()))
                ax[r][1].set_title("ID: "+str(pred[r].detach().cpu().numpy()))
            fig.tight_layout()
            nrunpath = "./"+name_of_run+"/"
            if not os.path.exists(nrunpath):
               os.makedirs(nrunpath)
            plt.savefig(nrunpath+str(epoch)+"_"+str(i)+".png")
            plt.close()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_ID {loss_id.val:.4f} ({loss_id.avg:.4f})\t'
                  'ID_ACC {id_acc:.4f}\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, id_acc=id_acc,loss_id=losses_id))

            with open(name_of_run+".txt", "a") as file_object:
            # Append 'hello' at the end of file
                file_object.write('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_ID {loss_id.val:.4f} ({loss_id.avg:.4f})\t'
                      'ID_ACC {id_acc:.4f}\n'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, id_acc=id_acc,loss_id=losses_id))

    loss_id_global_train.append(losses_id.avg)
    acc_id_global_train.append(id_acc)


def validate(valid_loader, model_id, criterion_id, epoch, name_of_run,num_classes):
    #Run one eval epoch
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_id = AverageMeter()

    correct = 0
    total = 0

    # switch to eval mode
    model_id.eval()

    end = time.time()
    for i, (input,bvp,id,gender,race,dir,frame0,lnd,mask) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        input = input.to(device=device, dtype=torch.float)
        id = id.to(device=device, dtype=torch.long)
        label = F.one_hot(id,num_classes=num_classes).float()
        with torch.no_grad():
            # compute output
            output = input
            t = 32
            out_id = model_id(normt(output[:,:,t,:,:]))
            pred = torch.argmax(out_id,dim=1)

            correct += ((pred == id).sum()).cpu().numpy()
            total += id.size(0)
            id_acc = correct/total

            loss_id = criterion_id(out_id,label)

            loss = loss_id
            loss = loss.float()

            # measure accuracy and record loss
            losses_id.update(loss_id.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % 50 == 0 or i == len(valid_loader)-2:
            fig,ax = plt.subplots(BATCH_SIZE,2,figsize=(15,15))
            for r in range(0,BATCH_SIZE):
                ax[r][0].imshow(input[r,:,32,:,:].permute(1,2,0).detach().cpu().numpy())
                ax[r][1].imshow(output[r,:,32,:,:].permute(1,2,0).detach().cpu().numpy())
                ax[r][0].set_title("ID: "+str(id[r].detach().cpu().numpy()))
                ax[r][1].set_title("ID: "+str(pred[r].detach().cpu().numpy()))
            fig.tight_layout()
            nrunpath = "./"+name_of_run+"/"
            if not os.path.exists(nrunpath):
               os.makedirs(nrunpath)
            plt.savefig(nrunpath+"Valid_"+str(epoch)+"_"+str(i)+".png")
            plt.close()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_ID {loss_id.val:.4f} ({loss_id.avg:.4f})\t'
                  'ID_ACC {id_acc:.4f}\n'.format(
                      epoch, i, len(valid_loader), batch_time=batch_time,
                      data_time=data_time, id_acc=id_acc,loss_id=losses_id))

            with open(name_of_run+".txt", "a") as file_object:
            # Append 'hello' at the end of file
                file_object.write('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_ID {loss_id.val:.4f} ({loss_id.avg:.4f})\t'
                      'ID_ACC {id_acc:.4f}\n'.format(
                          epoch, i, len(valid_loader), batch_time=batch_time,
                          data_time=data_time, id_acc=id_acc,loss_id=losses_id))

    loss_id_global_valid.append(losses_id.avg)
    acc_id_global_valid.append(id_acc)

def load_state_dict_senet50(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))


BATCH_SIZE = 8
NUM_WORKERS = 4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cudnn.benchmark = True

whatmodel = sys.argv[1]
whatdb = sys.argv[2]



if whatdb == "pure":
    num_classes = 10

if whatmodel == "googlenet" or whatmodel == "densenet"  or whatmodel == "resnet"  or whatmodel == "facenet" or whatmodel == "senet":
    if whatmodel == "googlenet":
        model_id = models.googlenet(weights='GoogLeNet_Weights.DEFAULT')
        model_id.fc  = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1),
        )
    if whatmodel == "densenet":
        model_id = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        model_id.classifier  = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1),
        )
    if whatmodel == "resnet":
        model_id = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model_id.fc  = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1),
        )
    if whatmodel == "facenet":
        model_id = InceptionResnetV1(pretrained='vggface2',classify=True)
        model_id.logits  = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
        )
        for param in model_id.parameters():
            param.requires_grad = False

        for param in model_id.logits.parameters():
            param.requires_grad = True
    if whatmodel == "senet":
        model_id = senet50(num_classes=8631, include_top=True)
        fname = "senet50_scratch_weight.pkl"
        with open(fname, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')
        own_state = model_id.state_dict()
        for name, param in weights.items():
            if name in own_state:
                own_state[name].copy_(torch.from_numpy(param))
        model_id.avgpool  = nn.AvgPool2d(kernel_size=4,stride=1,padding=0)

        model_id.fc  = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1),
        )
        for param in model_id.parameters():
            param.requires_grad = False

        for param in model_id.fc.parameters():
            param.requires_grad = True

else:
    print("input the right model")

model_id.to(device)

transformz = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

pklfile = open("./folds/"+whatdb+"_folds_rppg.pkl",'rb')

train_samples = pickle.load(pklfile)
valid_samples = pickle.load(pklfile)

if whatdb == "pure":
    train_dataset = load_cropped_pure(data=train_samples,shuffle=True, Training = True, transform=transformz)
    valid_dataset = load_cropped_pure(data=valid_samples,shuffle=False, Training = False, transform=transformz)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

normt = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

criterion_id =  nn.CrossEntropyLoss()
criterion_id = criterion_id.to(device)

optimizer = op.AdamW(model_id.parameters(), eps=1e-8, betas=(0.9, 0.999),lr=1e-4, weight_decay=0.05) #from SWIN paper change to resnet optimizer
name_of_run = "tight_"+whatmodel+"_"+whatdb+"_id"

for epoch in range(0, 1):
    train(train_loader, model_id, criterion_id,optimizer,epoch,name_of_run,num_classes)
    validate(valid_loader, model_id, criterion_id,epoch,name_of_run,num_classes)
    print(name_of_run)
    torch.save(model_id.state_dict(), name_of_run+".pt")

    fig,ax = plt.subplots(1,1)
    plt.plot(loss_id_global_train,'y-')
    plt.plot(loss_id_global_valid, 'b-')
    plt.grid()
    plt.savefig(name_of_run+"_id_loss")
    plt.close()

    fig,ax = plt.subplots(1,1)
    plt.plot(acc_id_global_train,'y-')
    plt.plot(acc_id_global_valid, 'b-')
    plt.grid()
    plt.savefig(name_of_run+"_id_acc")
    plt.close()
