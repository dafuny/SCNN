#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:14:41 2019

@author: luoyao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:37:03 2019

@author: luoyao
"""

import os.path as osp
from pathlib import Path
import numpy as np
import time
import logging

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#import resnet50
import torch.backends.cudnn as cudnn

from siamese_utils import SiaTrainDataset, ToTensor, Normalize
#from siamese_utils import
from io_utils import mkdir
#from wpdc_loss import WPDCLoss
import matplotlib.pylab as plt
import torch.nn.functional as F
from sia_loss import WPDCLoss, WPDCLoss_re,WPDCLoss_0, WPDCLoss_1, WPDCLoss_3

#global configuration
lr_iden = None
lr_wpdc = None
#arch
start_epoch = 1
param_fp_train='./train.configs/param_all_norm.pkl'
param_fp_val='./train.configs/param_all_norm_val.pkl'
warmup = 5
#opt_style 
batch_size = 64
base_lr_wpdc = 0.001
base_lr_iden = 0.00001
lr_iden = base_lr_iden
lr_wpdc = base_lr_wpdc
momentum = 0.9
weight_decay = 5e-4
epochs = 50
milestones = 30, 40
print_freq = 50
devices_id = [0]
workers = 8
filelists_train = "./label_train_aug_120x120.list.train"
filelists_val = "./label_train_aug_120x120.list.val"
root = "/home/luoyao/Project_3d/3D_face_solution/3DDFA_TPAMI/3DDFA_PAMI/train_aug_120x120" 
pre_model_params_path = "./training_debug/shp-constrain+wpdc-model/_checkpoint_epoch_50.pth.tar"
log_file = "./training_debug/logs/wpdc+identification_0/"
#loss
snapshot = "./training_debug/logs/wpdc+identification_0/"
log_mode = 'w'
resume = ''
size_average = True
num_classes = 62
frozen = 'false'
task = 'all'
test_initial = False
resample_num = 132

mkdir(snapshot)

class sia_net(nn.Module):
    def __init__(self , model):
        super(sia_net, self).__init__()
        #取掉model的后两层
        self.fc1 = nn.Sequential(
                nn.Sequential(*list(model.children())[:-2]),
                nn.AdaptiveAvgPool2d(1))

#        self.relu = nn.ReLU(inplace=True)
        self.fc1_0 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Linear(1024, 512))

        self.fc1_1 = nn.Sequential(
               nn.Linear(2048, 62))
        
    def forward_once(self, x):
        x = self.fc1(x)

        x = x.view(x.size()[0], -1) 
 
        feature = self.fc1_0(x)     #feature

#        feature = self.relu(feature)

        param = self.fc1_1(x)

        return feature, param
    
    def forward(self, input_l, input_r):
        feature_l, param_l = self.forward_once(input_l)
        feature_r, param_r = self.forward_once(input_r)

        return feature_l, feature_r, param_l, param_r

def show_plot(iteration, loss):
    plt.clf()
    plt.ion()
    plt.figure(1)
    plt.plot(iteration, loss,'-r')
    plt.draw()
    time.sleep(0.01)

def adjust_lr_iden(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    global lr_iden
    lr_iden = base_lr
    for param_group in optimizer.param_groups:
        lr_iden = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
        param_group['lr'] = lr_iden

def adjust_lr_wpdc(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    global lr_wpdc
    lr_wpdc = base_lr
    for param_group in optimizer.param_groups:
        lr_wpdc = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
        param_group['lr'] = lr_wpdc

def load_resnet50():
    model = torchvision.models.resnet50()
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2048, num_classes)
    return model

def load_resnet50_re():
    resnet = torchvision.models.resnet50()
    model = sia_net(resnet)
    
    return model

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint fo {filename}')



def train(train_loader, model, criterion_wpdc, criterion_iden, optimizer_wpdc, optimizer_iden, epoch):
    #status:training!
    model.train()
  
    for i, (img_l, img_r, label, target_l, target_r) in enumerate(train_loader):
        
        target_l.requires_grad = False
        target_r.requires_grad = False
        
        label.requires_grad = False
        label = label.cuda(non_blocking = True)
        
        target_l = target_l.cuda(non_blocking=True)
        target_r = target_r.cuda(non_blocking=True)
        
        feature_l, feature_r, output_l, output_r = model(img_l, img_r)

        #wpdc-constrain
        loss_wpdc = criterion_wpdc(output_l, output_r, label, target_l, target_r)
        optimizer_wpdc.zero_grad()
        loss_wpdc.backward(retain_graph=True)
        optimizer_wpdc.step()

        #shp-constrain
        loss_iden = criterion_iden(feature_l, feature_r, label)
        optimizer_iden.zero_grad()
        loss_iden.backward()
        optimizer_iden.step()
        
        loss = loss_wpdc+loss_iden;

        #log
        if i % epochs == 0:
            print('[Step:%d | Epoch:%d], lr_iden:%.8f, lr_wpdc:%.8f, loss_iden:%.8f, loss_wpdc:%.8f, loss:%.8f' % (i, epoch, lr_iden, lr_wpdc, loss_iden.data.cpu().numpy(), loss_wpdc.data.cpu().numpy(), loss.data.cpu().numpy()))
            print('[Step:%d | Epoch:%d], lr_iden:%.8f, lr_wpdc:%.8f, loss_iden:%.8f, loss_wpdc:%.8f, loss:%.6f' % (i, epoch, lr_iden, lr_wpdc, loss_iden.data.cpu().numpy(), loss_wpdc.data.cpu().numpy(), loss.data.cpu().numpy()), file=open(log_file + 'contrastive_print.txt','a'))

def validate(val_loader, model, criterion, epoch):
    
    model.eval()
    
    with torch.no_grad():
        losses = []

        for i, (input_l, input_r, label, target_l, target_r) in enumerate(val_loader):
            
            target_l.requires_grad = False
            target_r.requires_grad = False
            
            target_l = target_l.cuda(non_blocking=True)
            target_r = target_r.cuda(non_blocking=True)
            
            label.requires_grad = False
            label = label.cuda(non_blocking=True)
            
            feature_l, output_l = model(input_l)
            feature_r, output_r = model(input_r)
            
            loss = criterion(output_l, output_r, feature_l, feature_r, label, target_l, target_r)
            loss_cpu = loss.cpu()
            losses.append(loss_cpu.numpy())
            
            #show plot
            
        loss = np.mean(losses)
        print('Testing======>>>[Epoch:%d], loss:%.4f' % (epoch, loss))
        print('[Epoch:%d], loss:%.4f' % (epoch, loss), file=open(log_file + 'test_loss.txt','a'))
        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                     f'Loss {loss:.4f}\t')
        

def extract_param(model_params_path):
     map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
     checkpoint = torch.load(model_params_path,map_location=map_location)['state_dict']
     return  checkpoint
     
def main():
    
    #step1:define the model structure
#    model = load_resnet50()
    model = load_resnet50_re()
    torch.cuda.set_device(devices_id[0])
    model = nn.DataParallel(model, device_ids=devices_id).cuda()
    
    #load pre-train model param
#    checkpoint = extract_param(pre_model_params_path)
#    model.load_state_dict(checkpoint)

    #step2: loss and optimization method
    #criterion = ContrastiveLoss().cuda()
#    criterion = WPDCLoss().cuda()
#    criterion = WPDCLoss_re().cuda()
    criterion_wpdc = WPDCLoss_0().cuda()
    criterion_iden = WPDCLoss_3().cuda()
    optimizer_wpdc = torch.optim.SGD(model.parameters(),
                                lr = base_lr_wpdc,
                                momentum = momentum,
                                weight_decay = weight_decay,
                                nesterov = True)
    optimizer_iden = torch.optim.SGD(model.parameters(),
                                lr = base_lr_iden,
                                momentum = momentum,
                                weight_decay = weight_decay,
                                nesterov = True)
#    optimizer = torch.optim.Adam(model.parameters(),lr = base_lr)
    
    #step3:data
    normalize = Normalize(mean=127.5, std=128)
    
    train_dataset = SiaTrainDataset(
            root = root,
            filelists = filelists_train,
            param_fp = param_fp_train,
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            )
    val_dataset = SiaTrainDataset(
            root = root,
            filelists = filelists_val,
            param_fp = param_fp_val,
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            )
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers,
                            shuffle=False, pin_memory=True)
    
    cudnn.benchmark = True
    
    for epoch in range(start_epoch, epochs+1):
        #adjust learning rate
        #adjust_learning_rate(optimizer, epoch, milestones)
        adjust_lr_wpdc(optimizer_wpdc, base_lr_wpdc, epoch, epochs, 30)
        adjust_lr_iden(optimizer_iden, base_lr_iden, epoch, epochs, 30)
        #train for one epoch
        train(train_loader, model, criterion_wpdc, criterion_iden, optimizer_wpdc, optimizer_iden, epoch)
        #save model paramers
        filename = f'{snapshot}_checkpoint_epoch_{epoch}.pth.tar'
        save_checkpoint(
                {
                        'epoch':epoch,
                        'state_dict':model.state_dict()
                },
                filename
                )
        
#        validate(val_loader, model, criterion, epoch)


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
         bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


        
        
if __name__ == "__main__":

    main()

    ## observe model structure
#    model = load_resnet50()


    ## abserve batch sample data
#    train_dataset = SiaTrainDataset(
#        root = root,
#        filelists = filelists_train,
#        param_fp = param_fp_train,
#        transform = transforms.Compose([transforms.ToTensor()])
#        )
#    
#    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=workers,
#                              shuffle=False, pin_memory=True, drop_last=True)
#    dataiter = iter(train_loader)
#    
#    example_batch = next(dataiter)
#    concatenated = torch.cat((example_batch[0],example_batch[1]), 0)
#    imshow(torchvision.utils.make_grid(concatenated))
#    print(example_batch[2].numpy())