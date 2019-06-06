#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:46:07 2019

@author: luoyao
"""

import os
import os.path as osp
from pathlib import Path
import numpy as np 

import torch
import torch.utils.data as data
from PIL import Image
import pickle
import argparse
from io_utils import _numpy_to_tensor, _load_cpu, _load_gpu
from params import *
from collections import defaultdict
import random

def reconstruct_vertex(param, whitening=True, dense=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp"""
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))         #qian 12 dian ying gai shi u_base,concatenate() shi xiang pin jie yi ge 62 wei de shu zu
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean       #   x-mean/std ~ N(0,1)   you he yong ?????
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]        #suo you hang,qian 3 lie
    offset = p_[:, -1].reshape(3, 1)     #suo you hang ,dao shu di yi lie, tai sao le,bu neng ren!!!
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)

    # Note @yun suan fu shi dui ying yuan su xiang cheng!
    # dense and non-dense difference main display :w_shp:159645x40; w_shp:204x40
    if dense:      
        vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
    else:                                                          # reshape(3, -1, order='F') order='F'竖着读，竖着写，优先读/写一列
        """For 68 pts"""  # get 68 keypoint 3d position  p:3x3 (u + w_shp...):159645x1--->3x53215
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset   
        # for landmarks
        vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex

def reconstruct_vertex_shp(param, whitening=True, dense=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp"""
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))         #qian 12 dian ying gai shi u_base,concatenate() shi xiang pin jie yi ge 62 wei de shu zu
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean       #   x-mean/std ~ N(0,1)   you he yong ?????
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]        #suo you hang,qian 3 lie
    offset = p_[:, -1].reshape(3, 1)     #suo you hang ,dao shu di yi lie, tai sao le,bu neng ren!!!
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)

    # Note @ yun suan fu shi dui ying yuan su xiang cheng!
    # dense and non-dense difference main display :w_shp:159645x40; w_shp:204x40
    if dense:      
        vertex =  (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
    else:                                                          # reshape(3, -1, order='F') order='F'竖着读，竖着写，优先读/写一列
        """For 68 pts"""  # get 68 keypoint 3d position  p:3x3 (u + w_shp...):159645x1--->3x53215
        vertex =  (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F')
        # for landmarks
        vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex

def create_label_dict(path):
    label_dict = defaultdict(list)
    names_list = Path(path).read_text().strip().split('\n')
    for f_name in names_list:
        f_s = f_name.split('\000')
        label_dict[int(f_s[1])].append(f_s[0])
    
    return label_dict

def split_label(path):
    names_list = Path(path).read_text().strip().split('\n')
    img_name_nlabel = []
    for img_name in names_list:
        img_name_nlabel.append(img_name.split('\000')[0])
        
    return img_name_nlabel

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def _parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def read_pairs_ddfa(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    

class ToTensor(object):          #numpy-->torch
    def __call__(self, pic):             #shi yong __call__() ke yi zhi jie dui shi li jin xing diao yong,that is to say we can use instace get method
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor

class SiaTrainDataset(data.Dataset):
    def __init__(self, root, filelists, param_fp, transform=None, **kargs):
        self.root = root
        self.transform = transform
        self.label_dict = create_label_dict(filelists)    #fang bian make label
        self.lines = split_label(filelists)
        self.params = _numpy_to_tensor(_load_cpu(param_fp))
        self.img_loader = Image.open
    
    def _target_loader(self, index):
        target = self.params[index]
        
        return target
        
    def __getitem__(self, index):
        #random choose person1~personN
        label_1 = random.choice( range(len(self.label_dict)) )
        img1_name = random.choice( self.label_dict[label_1] )
        
        # %50 for same, %50 for diff 0:indicates diff,1:indicates same
        #is_same = random.randint(0,1)    
        # 100% probability
        #is_same = 1    #constrict the same people
        # 60% for different, 40% for same
        is_same = np.random.choice([0,1], p=[0.6, 0.4])
        
        if is_same:
            img2_name = random.choice(self.label_dict[label_1])
        else:
            while True:
                label_2 = random.choice( range(len(self.label_dict)) )
                if label_2 != label_1:
                    break
            img2_name = random.choice( self.label_dict[label_2] )
        
        img1_path = osp.join(self.root, img1_name)
        img2_path = osp.join(self.root, img2_name)
        
        img1 = self.img_loader(img1_path)
        img2 = self.img_loader(img2_path)
        
        index1 = self.lines.index(img1_name)
        index2 = self.lines.index(img2_name)
#        print(f'index1, index2 : {index1}, {index2}')
        target1 = self._target_loader(index1)
        target2 = self._target_loader(index2)
         
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.from_numpy(np.array([is_same], dtype = np.float32)), target1, target2
    
    def __len__(self):
        return len(self.lines)


class SiaTestDataset(data.Dataset):
    def __init__(self, filelists, root='', transform=None):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.img_loader = Image.open

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = self.img_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.lines)

class LFW_Pairs_Dataset(data.Dataset):
    def __init__(self, lfw_dir, pairs_txt, transform=None):
        self.transform = transform
        self.pairs = read_pairs(pairs_txt)
        self.lfw_dir = lfw_dir
        self.img_loader = Image.open

    def __getitem__(self, index):
        issame = None
        pair = self.pairs[index]
        if len(pair) == 3:
            path0 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(self.lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            img0 = self.img_loader(path0)
            img1 = self.img_loader(path1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, issame
     
    def __len__(self):
        return len(self.pairs)


class DDFA_Pairs_Dataset(data.Dataset):
    def __init__(self, root, pairs_txt, transform=None):
        self.transform = transform
        self.pairs = read_pairs_ddfa(pairs_txt)
        self.root = root
        self.img_loader = Image.open

    def __getitem__(self, index):

        pair = self.pairs[index]
        img0_name = pair[0]
        img1_name = pair[1]
        if int(pair[2]):
            issame = True
        else:
            issame = False
        path0 = os.path.join(self.root, img0_name)
        path1 = os.path.join(self.root, img1_name)

        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            img0 = self.img_loader(path0)
            img1 = self.img_loader(path1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, issame

    def __len__(self):
        return len(self.pairs)

class DDFATestDataset(data.Dataset):
    def __init__(self, filelists, root='', transform=None):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.img_loader = Image.open

    def __getitem__(self, index):                          #redefine function __getitem__
        path = osp.join(self.root, self.lines[index])
        img = self.img_loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):                                #redefine function __getitem
        return len(self.lines)
    

#test target
#if __name__ == "__main__":
#    root = "/home/luoyao/Project_3d/3D_face_solution/3DDFA_TPAMI/3DDFA_PAMI/train_aug_120x120" 
#    param_fp = './train.configs/param_all_norm.pkl'
#    label_path = "./label_train_aug_120x120.list.train"
#    
#    dataset = SiaTrainDataset(root,label_path, param_fp)
#
#    ulabel_path = "./train_aug_120x120.list.train"
#    wpdc_lines = Path(ulabel_path).read_text().strip().split('\n')
#    sia_lines = dataset.lines
#    
#    right_num = 0
#    for i, name in enumerate(sia_lines):
#        if i == wpdc_lines.index(name):
#            right_num = right_num + 1
#    print(f'right number: {right_num}')






































