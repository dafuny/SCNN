#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from math import sqrt
from io_utils import _load, _numpy_to_cuda, _numpy_to_tensor, _load_gpu
from params import *
import torch.nn.functional as F
from torch.autograd import Variable
import math

_to_tensor = _numpy_to_cuda  # gpu


def _parse_param_batch(param):       #from paramer p get  
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)   # p_:3x4
    p = p_[:, :, :3]    # qu qian 3 lie
    offset = p_[:, :, -1].view(N, 3, 1)      #qu dao shu di yi lie, may be this is the translation vector:x,y,z
    alpha_shp = param[:, 12:52].view(N, -1, 1)
    alpha_exp = param[:, 52:].view(N, -1, 1)
    return p, offset, alpha_shp, alpha_exp


class WPDCLoss_0(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132, margin=5):
        super(WPDCLoss_0, self).__init__()
        self.opt_style = opt_style
        self.param_mean = _to_tensor(param_mean)
        self.param_std = _to_tensor(param_std)

        self.u = _to_tensor(u)
        self.w_shp = _to_tensor(w_shp)
        self.w_exp = _to_tensor(w_exp)
        self.w_norm = _to_tensor(w_norm)

        self.w_shp_length = self.w_shp.shape[0] // 3  #53215
        self.keypoints = _to_tensor(keypoints)    # 68 keypoints ==> 68*3=204
        self.resample_num = resample_num
        self.margin = margin

    def reconstruct_and_parse(self, input, target):
        # reconstruct
        param = input * self.param_std + self.param_mean     #a*b dui ying yuan su xiang cheng
        param_gt = target * self.param_std + self.param_mean

        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)     #input
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)     #target

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def _calc_weights_resample(self, input_, target_):
        # resample index
        if self.resample_num <= 0:
            keypoints_mix = self.keypoints
        else:
            index = torch.randperm(self.w_shp_length)[:self.resample_num].reshape(-1, 1)    #torch.randperm()给定参数n，返回一个从0 到n -1 的随机整数排列
            keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1).cuda()  #torch.cat是将两个张量（tensor）拼接在一起,按维数1（列）拼接
            keypoints_mix = torch.cat((self.keypoints, keypoints_resample))                   #view(-1):jiang duo wei zhan kai cheng 1 dim
        w_shp_base = self.w_shp[keypoints_mix]                                              # 3*index+1or2 bu hui yue jie ma????
        u_base = self.u[keypoints_mix]
        w_exp_base = self.w_exp[keypoints_mix]

#        input = torch.tensor(input_.data.clone(), requires_grad=False)    #convert to tensor
#        target = torch.tensor(target_.data.clone(), requires_grad=False)
        input = input_.clone().detach().requires_grad_(False)
        target = target_.clone().detach().requires_grad_(False)

        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)   # fen bie fan hui bu tong bu fen de can shu

        input = self.param_std * input + self.param_mean
        target = self.param_std * target + self.param_mean

        N = input.shape[0]

        offset[:, -1] = offsetg[:, -1]       # ti huan zui hou yi ge yuan su, that is to say: ba z axis de yuan su ti huan diao!
        # zhe shi zai zuo bi ba?? ba ground truth de offsetg na guo lai zhi jie ti huan???
        weights = torch.zeros_like(input, dtype=torch.float)  #返回一个全0的Tensor，其维度与input相一致
        tmpv = (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).view(N, -1, 3).permute(0, 2, 1)   # chuan shuo zhong de s, size:[100,3,200]
         #permute():  将tensor的维度换位,比如图片img的size比如是（28，28，3）就可以利用img.permute(2,0,1)得到一个size为（3，28，28）的tensor                                                                          
        #use alpha_shp,alpha_exp and u_base reconstruct 3d face
        tmpv_norm = torch.norm(tmpv, dim=2)   # for dim=2 ,get 2 norm,   tmpv_norm: 100x3
        offset_norm = sqrt(w_shp_base.shape[0] // 3)  # //: 整数除法,返回不大于结果的一个最大的整数 600//3 = 200 keypoints

        # for pose
        param_diff_pose = torch.abs(input[:, :11] - target[:, :11])  #get qian 11 column, now! we do not care about z
        for ind in range(11):
            if ind in [0, 4, 8]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 0]        #wei he yao cheng zhe ge xi shu????
            elif ind in [1, 5, 9]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 1]
            elif ind in [2, 6, 10]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 2]
            else:    # 3, 7   indicates the x,y directino's translation vector
                weights[:, ind] = param_diff_pose[:, ind] * offset_norm
                

        ## This is the optimizest version
        # for shape_exp
        magic_number = 0.00057339936  # scale
        param_diff_shape_exp = torch.abs(input[:, 12:] - target[:, 12:])
        # weights[:, 12:] = magic_number * param_diff_shape_exp * self.w_norm
        w = torch.cat((w_shp_base, w_exp_base), dim=1)
        w_norm = torch.norm(w, dim=0)   
        # print('here')                     100x50              1x50 
        weights[:, 12:] = magic_number * param_diff_shape_exp * w_norm     #dui ying yuan su xiang cheng 
                                                                           #torch.mm: ju zhen xiang cheng
        eps = 1e-6
        weights[:, :11] += eps        # bi kai di 12 column
        weights[:, 12:] += eps

        # normalize the weights
        maxes, _ = weights.max(dim=1)   # mei yi hang de zui da zhi,bing fan hui gai lie de suo yin
        maxes = maxes.view(-1, 1)    #bian cheng n*1
        weights /= maxes

        # zero the z
        weights[:, 11] = 0    #jiang di 12 coluwn zhi wei 0   weights :100x62

        return weights

        
    def forward(self, input_l, input_r, label, target_l, target_r):
        if self.opt_style == 'resample':
            
            weights_l = self._calc_weights_resample(input_l, target_l)
            weights_r = self._calc_weights_resample(input_r, target_r)
           
            loss_wpdc = weights_l * (input_l-target_l)**2 + weights_r * (input_r-target_r)**2

            total_loss = loss_wpdc.mean()

            return total_loss
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')        


class WPDCLoss_1(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132, margin=5.0):
        super(WPDCLoss_1, self).__init__()
        self.opt_style = opt_style
        self.margin = margin

        
    def forward(self, input_l, input_r, label):
        if self.opt_style == 'resample':

            #shape contrain
            E_w_i = F.pairwise_distance(input_l[:,12:], input_r[:,12:], p=1, keepdim = True)

#            shp_constrain = label * (2/self.margin) * E_w_i**2 
            e = 2.71828
            shp_contrastive = label * (2/self.margin) * E_w_i**2 + (1-label)*2*self.margin*e**(-2.77*E_w_i/self.margin)
            
            total_loss = shp_contrastive.mean()
            return total_loss
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')     

class WPDCLoss_4(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132, margin=1.0):
        super(WPDCLoss_4, self).__init__()
        self.opt_style = opt_style
        self.margin = margin

        
    def forward(self, input_l, input_r, label):
        if self.opt_style == 'resample':

            #shape contrain
            shp_diff = F.pairwise_distance(input_l[:,12:], input_r[:,12:], p=2, keepdim = True)
#            shp_contrastive = 0.5 * (label) * torch.pow(shp_diff, 2) + (1-label) * torch.pow(torch.clamp(self.margin - shp_diff, min=0.0), 2)
            shp_contrastive = 0.5 * (label) * torch.pow(shp_diff, 2)
            total_loss = shp_contrastive.mean()

            return total_loss
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')     

#this loss is for constrain identification
class WPDCLoss_2(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132, margin=5):
        super(WPDCLoss_2, self).__init__()
        self.opt_style = opt_style
        self.margin = margin

    def forward(self, feature_l, feature_r, label):
        if self.opt_style == 'resample':
            #feature contrain
            E_w_f = F.pairwise_distance(feature_l, feature_r, p=1, keepdim = True)

            Q = self.margin
            e = 2.71828
            iden_contrastive = label * (2/Q) * E_w_f**2 + (1-label)*2 * Q * e**(-2.77*E_w_f/Q)

            total_loss = iden_contrastive.mean()
            return total_loss
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')

class WPDCLoss_3(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132, margin=1.0):
        super(WPDCLoss_3, self).__init__()
        self.opt_style = opt_style
        self.margin = margin

    def forward(self, feature_l, feature_r, label):
        if self.opt_style == 'resample':
            #feature contrain
            diff_iden = F.pairwise_distance(feature_l, feature_r, p=2, keepdim = True)
            iden_contrastive = 0.5 * (label) * torch.pow(diff_iden, 2) + (1-label) * torch.pow(torch.clamp(self.margin - diff_iden, min=0.0), 2)

#            diff_iden = torch.pow(feature_l-feature_r, 2)
#            gen_iden = torch.sum(diff_iden, 1, True)
#            impos = torch.sqrt(gen_iden+1e-6)
#            iden_contrastive = 0.5 * (label) * gen_iden + (1-label) * torch.pow(torch.clamp(self.margin - impos, min=0.0), 2)

            total_loss = iden_contrastive.mean()
            return total_loss
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')



#if __name__ == "__main__":
#    wpdc = WPDCLoss()
