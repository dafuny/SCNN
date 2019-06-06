#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:15:39 2019

@author: luoyao
"""

#!/usr/bin/env python3
# coding: utf-8

import torchvision
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
import numpy as np


from siamese_utils import Normalize, SiaTestDataset, reconstruct_vertex, reconstruct_vertex_shp

import scipy.io as sio
import os.path as osp
import os
from io_utils import _load, _dump, mkdir

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

def load_resnet50():
    resnet = torchvision.models.resnet50()
    model = sia_net(resnet)

    return model


def sia_extract_param(checkpoint_fp, root = '', filelists = None, device_ids = [0],
                      batch_size = 128, num_workers = 8):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']     ## 把张量从GPU 0~7 移动到 GPU 0, get paramerm's weight
    torch.cuda.set_device(device_ids[0])                                                #bing cong zi dian zhong na chu key=='state_dict' de nei rong
    model = load_resnet50()    #get a model explain or document
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint)
    dataset = SiaTestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([transforms.ToTensor(), Normalize(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cudnn.benchmark = True   #总的来说，大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    model.eval()   #jiang model bian cheng test pattern

    end = time.time()
    param_62d = []
    feature_512d = []

    with torch.no_grad():     #bu ji suan gradient
        for _, inputs in enumerate(data_loader):
            inputs = inputs.cuda()           #fang dao gpu shang jin xing yun suan 
            feature, _, param, _ = model(inputs, inputs)

                
            for i in range(param.shape[0]):     #output.shape[0] = 128 == batch_size
                param_gen = param[i].cpu().numpy().flatten()
                feature_gen = feature[i].cpu().numpy().flatten()

                param_62d.append(param_gen)
                feature_512d.append(feature_gen)
                
        param_62d = np.array(param_62d, dtype=np.float32)    # from list convert to array
        feature_512d = np.array(feature_512d, dtype=np.float32)

    print(f'Extracting params take {time.time() - end: .3f}s')
    
    return feature_512d, param_62d


#def sia_extract_param_lfw(checkpoint_fp, filelists = None, device_ids = [0],
#                      batch_size = 128, num_workers = 8):
#    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
#    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']     ## 把张量从GPU 0~7 移动到 GPU 0, get paramerm's weight
#    torch.cuda.set_device(device_ids[0])                                                #bing cong zi dian zhong na chu key=='state_dict' de nei rong
#    model = load_resnet50()    #get a model explain or document
#    model = nn.DataParallel(model, device_ids=device_ids).cuda()
#    model.load_state_dict(checkpoint)
#    dataset = LFW_Dataset(filelists=filelists,
#                              transform=transforms.Compose([transforms.ToTensor(), Normalize(mean=127.5, std=128)]))
#    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#    cudnn.benchmark = True   #总的来说，大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
#    model.eval()   #jiang model bian cheng test pattern
#
#    end = time.time()
#    param_62d = []
#    feature_512d = []
#
#    with torch.no_grad():     #bu ji suan gradient
#        for _, inputs in enumerate(data_loader):
#            inputs = inputs.cuda()           #fang dao gpu shang jin xing yun suan 
#            feature, _, param, _ = model(inputs, inputs)
#
#                
#            for i in range(param.shape[0]):     #output.shape[0] = 128 == batch_size
#                param_gen = param[i].cpu().numpy().flatten()
#                feature_gen = feature[i].cpu().numpy().flatten()
#
#                param_62d.append(param_gen)
#                feature_512d.append(feature_gen)
#                
#        param_62d = np.array(param_62d, dtype=np.float32)    # from list convert to array
#        feature_512d = np.array(feature_512d, dtype=np.float32)
#
#    print(f'Extracting params take {time.time() - end: .3f}s')
#    
#    return feature_512d, param_62d

def extract_3DMM(data_info):
    
    _, param_62d = sia_extract_param(data_info['checkpoint_fp'], data_info['root'], data_info['filelists_test'])

    return param_62d


def benchmark_3d_vertex_shp(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex_shp(params[i])
        outputs.append(lm)
    return outputs

def benchmark_3d_vertex(params, dense = True):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i], dense)
        outputs.append(lm)
    return outputs

def benchmark_3d_vertex_save(params, img_names_list, method='', dense = True):
    save_path = 'result/'+method+'/'
    mkdir(save_path)
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i], dense = True)
        fn = img_names_list[i]
        wfp = osp.join(save_path, fn.replace('.jpg', '.mat'))
        print(wfp)
        sio.savemat(wfp, {'vertex': lm})

def benchmark_3d_vertex_shp_save(params, img_names_list, method='', dense = True):
    save_path = 'result-no-pose/'+method+'/'
    mkdir(save_path)
    for i in range(params.shape[0]):
        lm = reconstruct_vertex_shp(params[i], dense = True)
        fn = img_names_list[i]
        wfp = osp.join(save_path, fn.replace('.jpg', '.mat'))
        print(wfp)
        sio.savemat(wfp, {'vertex': lm})

def feature_512d_save(feature_512d, img_name_list):
    for i in range(feature_512d.shape[0]):
        feature = feature_512d[i]
        fn = img_name_list[i]
        dirname, basename = os.path.split(fn) 
        dirname.replace('align_lfw', 'feature_lfw')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        wfp = osp.join(dirname, basename.replace('.png', '.npy'))
        _dump(wfp, feature)
































