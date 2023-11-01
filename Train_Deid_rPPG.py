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
from load_cropped_blocks_rppg_pure_withphysgt import load_cropped_pure
from torch.utils.data import DataLoader
from utils.Average_Accuracy import AverageMeter
from utils.Average_Accuracy import Acc
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as op
import time
import random
from scipy.fft import fft
from scipy import signal
import math
from AutoEncoder_3d import AutoEncoder
import os
import torch.backends.cudnn as cudnn
import configparser
import sys
from datetime import datetime
from pytorch_msssim import ssim
from Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from utils.signals_stuff import butter_bandpass, norm, hr_fft,NegPearson
from PhysNet import PhysNet as PN
from utils.utils_chrom import gpu_CHROM, raw_rppg
from facenet_pytorch import InceptionResnetV1
from senet import senet50

loss_global_train = []
loss_global_valid = []
loss_rppg_global_train = []
loss_rppg_global_valid = []
loss_rec_global_train = []
loss_rec_global_valid = []
loss_id_global_train = []
loss_id_global_valid = []
acc_id_global_train = []
acc_id_global_valid = []
acc_id_global_train = []
acc_id_global_valid = []

def train(train_loader, model_id,model_autoenc,model_rppg, criterion_id, criterion_rec,criterion_rppg, optimizer_g, epoch, name_of_run,isrppg,num_classes,whatmodelrppg):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_rppg = AverageMeter()
    losses_adv_g = AverageMeter()
    losses_rec = AverageMeter()
    losses_id = AverageMeter()
    confidence = AverageMeter()
    acc = Acc()

    # switch to train mode
    model_autoenc.train()
    model_id.eval()
    if isrppg and (whatmodelrppg == "physformer" or whatmodelrppg == "physnet"):
        model_rppg.eval()

    correct = 0
    total = 0

    end = time.time()
    for i, (input,bvp,id,gender,race,dir,frame0,lnd,mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        id = id.to(device=device, dtype=torch.long)
        label = F.one_hot(id,num_classes=num_classes)
        label = label.to(device=device, dtype=torch.float)

        fake = model_autoenc(input)

        T = (input.size()[2])
        new_T = int(T/8)
        range_t = np.arange(0,T)
        np.random.shuffle(range_t)
        range_t = range_t[:new_T]
        lozzes_id = torch.zeros(new_T)
        c = 0
        for t in range_t:
            out_id = model_id(normt(fake[:,:,t,:,:]))
            pred = torch.argmax(out_id,dim=1)
            conf = torch.max(out_id,dim=1)[0]
            conf = np.mean(conf.detach().cpu().numpy())
            flat = (1/out_id.size(1))*torch.ones_like(out_id)
            lozzes_id[c] = criterion_id(out_id,label)-0.1*torch.mean((torch.sqrt(torch.sum(torch.square(out_id-flat),1))/(torch.sqrt(torch.sum(torch.square(out_id),1))+torch.sqrt(torch.sum(torch.square(flat),1)))))
            c+=1
            correct += ((pred == id).sum()).cpu().numpy()
            total += id.size(0)

        id_acc = correct/total
        g_loss_constraint = torch.mean(lozzes_id)

        loss_rec1 = (1 - ssim( input.permute(0,2,1,3,4).flatten(0,1), fake.permute(0,2,1,3,4).flatten(0,1), data_range=1, size_average=True))
        loss_rec2 = criterion_rec(input,fake)
        g_loss_rec = loss_rec1*0.5 + loss_rec2*0.5

        if isrppg:
            if whatmodelrppg == "chrom":
                mask = mask.to(device=device, dtype=torch.float)
                masked = fake*mask
                out_rppg = gpu_CHROM(raw_rppg(masked))
            if whatmodelrppg == "physformer":
                out_rppg = model_rppg(fake,2.0)
            if whatmodelrppg == "physnet":
                out_rppg = model_rppg(fake)

            out_rppg = (out_rppg - torch.mean(out_rppg)) / torch.std(out_rppg)
            bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize
            g_loss_rppg = criterion_rppg(out_rppg,bvp)

        lambda_rec = alpha
        lambda_constraint = beta
        lambda_rppg = gamma
        g_loss = lambda_rec * g_loss_rec + lambda_constraint * g_loss_constraint
        if isrppg:
            g_loss = g_loss + lambda_rppg*g_loss_rppg

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        if not isrppg:
            g_loss_rppg = g_loss
        losses.update(g_loss.item(), input.size(0))
        losses_rppg.update(g_loss_rppg.item(), input.size(0))
        losses_id.update(g_loss_constraint.item(), input.size(0))
        losses_rec.update(g_loss_rec.item(), input.size(0))
        confidence.update(conf, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        bs = input.size(0)
        if bs > 4:
            bs = 4
        if i == int(len(train_loader)/2) or i == len(train_loader)-2:
            print_string =  """Epoch: [{0}][{1}/{2}] \
Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \
Data {data_time.val:.3f} ({data_time.avg:.3f}) \
Loss {loss.val:.4f} ({loss.avg:.4f})\
Loss_ID {loss_id.val:.4f} ({loss_id.avg:.4f}) \
Loss_RPPG {loss_rppg.val:.4f} ({loss_rppg.avg:.4f}) \
Loss_REC {loss_rec.val:.4f} ({loss_rec.avg:.4f}) \
Conf {conf.avg:.4f} \
ID_ACC {id_acc:.4f} \n""".format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time, id_acc=id_acc,loss_id=losses_id,loss_rec=losses_rec,loss=losses,loss_rppg=losses_rppg,conf=confidence)
            print(print_string)

            with open(nrunpath+name_of_run+".txt", "a") as file_object:
            # Append 'hello' at the end of file
                file_object.write(print_string)

            fig,ax = plt.subplots(bs,2,figsize=(25,25))
            for r in range(0,bs):
                ax[r][0].imshow(input[r,:,t,:,:].permute(1,2,0).detach().cpu().numpy())
                ax[r][1].imshow(fake[r,:,t,:,:].permute(1,2,0).detach().cpu().numpy())
                ax[r][0].set_title("ID: "+str(id[r].detach().cpu().numpy()))
                ax[r][1].set_title("ID: "+str(pred[r].detach().cpu().numpy()))
            fig.tight_layout()
            if not os.path.exists(nrunpath+"plot/"):
               os.makedirs(nrunpath+"plot/")
            plt.savefig(nrunpath+"plot/"+"Train_"+str(epoch)+"_"+str(i)+".png")
            plt.close()

    loss_global_train.append(losses.avg)
    loss_rppg_global_train.append(losses_rppg.avg)
    loss_rec_global_train.append(losses_rec.avg)
    loss_id_global_train.append(losses_id.avg)
    acc_id_global_train.append(id_acc)

def validate(valid_loader, model_id,model_autoenc,model_rppg, criterion_id, criterion_rec, criterion_rppg, epoch, name_of_run,isrppg,num_classes,whatmodelrppg):

    #Run one eval epoch
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_rppg = AverageMeter()
    losses_rec = AverageMeter()
    losses_id = AverageMeter()
    confidence = AverageMeter()
    acc = Acc()
    correct = 0
    total = 0

    # switch to eval mode
    model_autoenc.eval()
    model_id.eval()
    if isrppg and (whatmodelrppg == "physformer" or whatmodelrppg == "physnet"):
        model_rppg.eval()

    correct = 0
    total = 0

    end = time.time()
    for i, (input,bvp,id,gender,race,dir,frame0,lnd,mask) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        input = input.to(device=device, dtype=torch.float)
        id = id.to(device=device, dtype=torch.long)
        label = F.one_hot(id,num_classes=num_classes)
        label = label.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        with torch.no_grad():
            fake = model_autoenc(input)
            T = (input.size()[2])
            new_T = int(T/8)
            range_t = np.arange(0,T)
            np.random.shuffle(range_t)
            range_t = range_t[:new_T]
            lozzes_id = torch.zeros(new_T)
            c = 0
            for t in range_t:
                out_id = model_id(normt(fake[:,:,t,:,:]))
                pred = torch.argmax(out_id,dim=1)
                conf = torch.max(out_id,dim=1)[0]
                conf = np.mean(conf.detach().cpu().numpy())
                flat = (1/out_id.size(1))*torch.ones_like(out_id)
                lozzes_id[c] = criterion_id(out_id,label)-0.1*torch.mean((torch.sqrt(torch.sum(torch.square(out_id-flat),1))/(torch.sqrt(torch.sum(torch.square(out_id),1))+torch.sqrt(torch.sum(torch.square(flat),1)))))
                c+=1
                correct += ((pred == id).sum()).cpu().numpy()
                total += id.size(0)

            id_acc = correct/total
            g_loss_constraint = torch.mean(lozzes_id)

            loss_rec1 = (1 - ssim( input.permute(0,2,1,3,4).flatten(0,1), fake.permute(0,2,1,3,4).flatten(0,1), data_range=1, size_average=True))
            loss_rec2 = criterion_rec(input,fake)
            g_loss_rec = loss_rec1*0.5 + loss_rec2*0.5
            if isrppg:
                if whatmodelrppg == "chrom":
                    mask = mask.to(device=device, dtype=torch.float)
                    masked = fake*mask
                    out_rppg = gpu_CHROM(raw_rppg(masked))
                if whatmodelrppg == "physformer":
                    out_rppg = model_rppg(fake,2.0)
                if whatmodelrppg == "physnet":
                    out_rppg = model_rppg(fake)
                out_rppg = (out_rppg - torch.mean(out_rppg)) / torch.std(out_rppg)
                bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize
                g_loss_rppg = criterion_rppg(out_rppg,bvp)

            lambda_rec = alpha#10 * -2/10 -> -2
            lambda_constraint = beta#10
            lambda_rppg = gamma#10
            g_loss = lambda_rec * g_loss_rec + lambda_constraint * g_loss_constraint
            if isrppg:
                g_loss = g_loss + lambda_rppg*g_loss_rppg
            if not isrppg:
                g_loss_rppg = g_loss

            losses.update(g_loss.item(), input.size(0))
            losses_rppg.update(g_loss_rppg.item(), input.size(0))
            losses_id.update(g_loss_constraint.item(), input.size(0))
            losses_rec.update(g_loss_rec.item(), input.size(0))
            confidence.update(conf, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        bs = input.size(0)

        if bs > 4:
            bs = 4
        if i == int(len(valid_loader)/2) or i == len(valid_loader)-2:
            print_string =  """Epoch: [{0}][{1}/{2}] \
Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \
Data {data_time.val:.3f} ({data_time.avg:.3f}) \
Loss {loss.val:.4f} ({loss.avg:.4f})\
Loss_ID {loss_id.val:.4f} ({loss_id.avg:.4f}) \
Loss_RPPG {loss_rppg.val:.4f} ({loss_rppg.avg:.4f}) \
Loss_REC {loss_rec.val:.4f} ({loss_rec.avg:.4f}) \
Conf {conf.avg:.4f} \
ID_ACC {id_acc:.4f} \n""".format(epoch, i, len(valid_loader), batch_time=batch_time,data_time=data_time, id_acc=id_acc,loss_id=losses_id,loss_rec=losses_rec,loss=losses,loss_rppg=losses_rppg,conf=confidence)
            print(print_string)
            with open(nrunpath+name_of_run+".txt", "a") as file_object:
            # Append 'hello' at the end of file
                file_object.write(print_string)

            fig,ax = plt.subplots(bs,2,figsize=(25,25))
            for r in range(0,bs):
                ax[r][0].imshow(input[r,:,t,:,:].permute(1,2,0).detach().cpu().numpy())
                ax[r][1].imshow(fake[r,:,t,:,:].permute(1,2,0).detach().cpu().numpy())
                ax[r][0].set_title("ID: "+str(id[r].detach().cpu().numpy()))
                ax[r][1].set_title("ID: "+str(pred[r].detach().cpu().numpy()))
            fig.tight_layout()
            if not os.path.exists(nrunpath+"plot/"):
               os.makedirs(nrunpath+"plot/")
            plt.savefig(nrunpath+"plot/"+"Valid_"+str(epoch)+"_"+str(i)+".png")
            plt.close()

    loss_global_valid.append(losses.avg)
    loss_rppg_global_valid.append(losses_rppg.avg)
    loss_rec_global_valid.append(losses_rec.avg)
    loss_id_global_valid.append(losses_id.avg)
    acc_id_global_valid.append(id_acc)


config = configparser.ConfigParser()
config.read(sys.argv[1])
alpha = float(sys.argv[2])
beta = float(sys.argv[3])
gamma = float(sys.argv[4])
whatmodelid = str(sys.argv[5])
whatmodelrppg = str(sys.argv[6])
suffix = sys.argv[7]

BATCH_SIZE = int(config['Params']['BATCH_SIZE'])
NUM_WORKERS = int(config['Params']['NUM_WORKERS'])
learn_rate = float(config['Params']['Learning_rate'])
total_epochs = int(config['Params']['Epochs'])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# set random number
random.seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# set the cudnn
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

whatdb = config['Data']['Dataset']

name_of_run = "autoid_"+whatmodelid+"_"+whatmodelrppg+"_"+whatdb+"_"+str(alpha)+"_"+str(beta)+"_"+str(gamma)+"_"+suffix
nrunpath = "./"+name_of_run+"/"
if not os.path.exists(nrunpath):
   os.makedirs(nrunpath)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
my_config_parser_dict = {s:dict(config.items(s)) for s in config.sections()}

with open(nrunpath+name_of_run+".txt", "a") as file_object:
# Append 'hello' at the end of file
    file_object.write(dt_string)
    file_object.write(str(my_config_parser_dict))

if whatdb == "pure":
    num_classes = 10

if whatmodelid == "googlenet" or whatmodelid == "densenet"  or whatmodelid == "resnet" or whatmodelid == "facenet"  or whatmodelid == "senet":
    if whatmodelid == "googlenet":
        model_id = models.googlenet(weights='GoogLeNet_Weights.DEFAULT')
        model_id.fc  = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1),
        )
        model_id.load_state_dict(torch.load("./trainedmodels/tight_googlenet_"+whatdb+"_id.pt"))

    if whatmodelid == "densenet":
        model_id = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        model_id.classifier  = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1),
        )
        model_id.load_state_dict(torch.load("./trainedmodels/tight_densenet_"+whatdb+"_id.pt"))

    if whatmodelid == "resnet":
        model_id = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model_id.fc  = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1),
        )
        model_id.load_state_dict(torch.load("./trainedmodels/tight_resnet_"+whatdb+"_id.pt"))

    if whatmodelid == "facenet":
        model_id = InceptionResnetV1(pretrained='vggface2',classify=True)
        model_id.logits  = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
        )
        model_id.load_state_dict(torch.load("./trainedmodels/tight_facenet_"+whatdb+"_id.pt"))
    if whatmodelid == "senet":
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
        model_id.load_state_dict(torch.load("./trainedmodels/tight_senet_"+whatdb+"_id.pt"))

else:
    print("input the right model")
    exit()

for param in model_id.parameters():
    param.requires_grad = False
model_id = nn.DataParallel(model_id)
model_id.to(device)

if whatmodelrppg == "none":
    model_rppg = "none"
    isrppg = False
if whatmodelrppg == "physformer":
    isrppg = True
    seq_len = 64
    model_rppg = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(seq_len,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    model_rppg = nn.DataParallel(model_rppg)
    model_rppg.load_state_dict(torch.load("Physformer_"+whatdb+".pt"))
    model_rppg = model_rppg.to(device)
    for param in model_rppg.parameters():
        param.requires_grad = False
if whatmodelrppg == "physnet":
    isrppg = True
    seq_len = 64
    model_rppg = PN(seq_len)
    model_rppg = nn.DataParallel(model_rppg)
    model_rppg.load_state_dict(torch.load("Physnet_"+whatdb+".pt"))
    model_rppg = model_rppg.to(device)
    for param in model_rppg.parameters():
        param.requires_grad = False
if whatmodelrppg == "chrom":
    model_rppg = "chrom"
    isrppg = True
    seq_len = 64

model_autoenc = AutoEncoder()
model_autoenc = nn.DataParallel(model_autoenc)
model_autoenc = model_autoenc.to(device)

transformz = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),
    ])

pklfile = open("./folds/"+whatdb+"_folds_rppg.pkl",'rb')

train_samples = pickle.load(pklfile)
valid_samples = pickle.load(pklfile)

if whatdb == "pure":
    train_dataset = load_cropped_pure(rppg_net=whatmodelrppg,data=train_samples,shuffle=True, Training = True, transform=transformz)
    valid_dataset = load_cropped_pure(rppg_net=whatmodelrppg,data=valid_samples,shuffle=False, Training = False, transform=transformz)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

normt = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

criterion_rec = nn.MSELoss()
criterion_id =  nn.CrossEntropyLoss()
criterion_rppg = NegPearson()

criterion_rec = criterion_rec.to(device)
criterion_id = criterion_id.to(device)
criterion_rppg = criterion_rppg.to(device)

optimizer_g = op.AdamW(model_autoenc.parameters(), eps=1e-8, betas=(0.9, 0.999),lr=learn_rate, weight_decay=0.05) #from SWIN paper change to resnet optimizer
for epoch in range(0, total_epochs):
    #if epoch > 4:
    #    beta = -1/10
    train(train_loader, model_id, model_autoenc,model_rppg, criterion_id, criterion_rec,criterion_rppg,optimizer_g,epoch,name_of_run,isrppg,num_classes,whatmodelrppg)
    validate(valid_loader, model_id, model_autoenc,model_rppg, criterion_id, criterion_rec,criterion_rppg,epoch,name_of_run,isrppg,num_classes,whatmodelrppg)
    torch.save(model_autoenc.state_dict(), name_of_run+".pt")

    fig,ax = plt.subplots(1,1)
    plt.plot(loss_global_train,'y-')
    plt.plot(loss_global_valid, 'b-')
    plt.grid()
    plt.savefig(nrunpath+"plot/"+name_of_run+"_loss.png")
    plt.close()

    fig,ax = plt.subplots(1,1)
    plt.plot(loss_rec_global_train,'y-')
    plt.plot(loss_rec_global_valid, 'b-')
    plt.grid()
    plt.savefig(nrunpath+"plot/"+name_of_run+"_rec_loss.png")
    plt.close()

    fig,ax = plt.subplots(1,1)
    plt.plot(loss_id_global_train,'y-')
    plt.plot(loss_id_global_valid, 'b-')
    plt.grid()
    plt.savefig(nrunpath+"plot/"+name_of_run+"_id_loss.png")
    plt.close()

    fig,ax = plt.subplots(1,1)
    plt.plot(acc_id_global_train,'y-')
    plt.plot(acc_id_global_valid, 'b-')
    plt.grid()
    plt.savefig(nrunpath+"plot/"+name_of_run+"_id_acc.png")
    plt.close()

    fig,ax = plt.subplots(1,1)
    plt.plot(loss_rppg_global_train,'y-')
    plt.plot(loss_rppg_global_valid, 'b-')
    plt.grid()
    plt.savefig(nrunpath+"plot/"+name_of_run+"_rppg_loss.png")
    plt.close()
