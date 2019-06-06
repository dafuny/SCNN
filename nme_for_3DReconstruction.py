#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:13:22 2019

@author: luoyao
"""

import numpy as np
from math import sqrt
from io_utils import _load, _dump, mkdir
from sia_extract_param import extract_3DMM, benchmark_3d_vertex_save, benchmark_3d_vertex_shp_save

import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict



checkpoint_fp_sia = "training_debug/logs/wpdc+shp+identification_4_V_lr05/_checkpoint_epoch_50.pth.tar"
checkpoint_fp_wpdc = "training_debug/logs/wpdc_alpha/_checkpoint_epoch_50.pth.tar"

root = "/home/luoyao/Project_3d/3D_face_solution/3DDFA_TPAMI/3DDFA_PAMI/train_aug_120x120" 
filelists_test = "./label_train_aug_120x120.list.val"
no_label_filelists_test = "./train_aug_120x120.list.val"
img_names_list = Path(no_label_filelists_test).read_text().strip().split('\n')

boxplot_save_path = 'training_debug/logs/joint-loss-boxplot-7/'

mkdir('result-no-pose/')
#mkdir(boxplot_save_path)

#62d param result path
param_fp_gt='./train.configs/param_all_norm_val.pkl' 
sia_param_62d_path = 'result-no-pose/sia_param_62d.npy' 
wpdc_param_62d_path = 'result-no-pose/wpdc_param_62d.npy' 


def read_line(path):
    img_names_list = Path(path).read_text().strip().split('\n')
    
    return img_names_list

def order_index_person(test_file_path):
    
    img_names_list = read_line(test_file_path)
    
    person_dict = defaultdict(list)
    
    for i, img_name in enumerate(img_names_list):
        person = int(img_name.split('\000')[1])
        person_dict[person].append(i)
        
    return person_dict

def extract_nme(person_dict, nme_list):
    person_nme_dic = defaultdict(list)
    
    for person, index_list in person_dict.items():
        for i in index_list:
            person_nme_dic[person].append(nme_list[i])
    
    return person_nme_dic

def box_plot(save_path, person_rme_dict1, person_rme_dict2):
    
    for name in person_rme_dict1.keys():
        #print(name)
        data = [person_rme_dict1[name], person_rme_dict2[name]]
        
        fig = plt.figure(figsize=(9,9),dpi=100)
        ax = fig.add_subplot(111)
        img_name = str(name)+'.jpg'
        ax.boxplot(data)
        ax.set_xticklabels([str(name)+'-our_method', str(name)+'-other_method'])
        
        ax.set_ylabel("diff-value")
        ax.set_xlabel("methods")
        fig.savefig(save_path+img_name)
        plt.close()

    # use 62d param reconstruct 3d vertex, consider pose, and save 'xxx.mat'
def param_recon_3d_vertex(param_path, reference_img_names_list, method='', dense=True):
    # load 62d param 
    param_62d = _load(param_path)
    # reconstruct 3d face shape, include pose information, and save 3d vertex info
    benchmark_3d_vertex_save(param_62d, reference_img_names_list, method, dense=True)

    # use 62d param reconstruct 3d vertex, do not consider pose, and save 'xxx.mat'
def param_recon_3d_vertex_shp(param_path, reference_img_names_list, method='', dense=True):
    # load 62d param 
    param_62d = _load(param_path)
    # reconstruct 3d face shape, no pose information, and save 3d vertex info
    benchmark_3d_vertex_shp_save(param_62d, reference_img_names_list, method, dense=True)

def extract_62d_param_save(info):
    sia_param_62d = extract_3DMM(info)
    _dump(info['save_path'], sia_param_62d)

def nan_pose_diff(gt_param, fit_param):
    pose_diff = gt_param[:,:12]-fit_param[:,:12]
    dis = np.sqrt(np.sum(np.power(pose_diff, 2), 1))
    
    return dis

if __name__ == '__main__':
    
    #person_dict = order_index_person(filelists_test)

    sia_info = {"checkpoint_fp":checkpoint_fp_sia, "root":root, "filelists_test":no_label_filelists_test, "save_path":sia_param_62d_path}
    wpdc_info = {"checkpoint_fp":checkpoint_fp_wpdc, "root":root, "filelists_test":no_label_filelists_test, "save_path":wpdc_param_62d_path}


    #extract 62d param and save it
#    extract_62d_param_save(sia_info)
#    extract_62d_param_save(wpdc_info)


    #it's inportant for param map img_names_list
    # reconstruct 3D dense face shape, include pose
#    param_recon_3d_vertex(sia_param_62d_path, img_names_list, method = 'vertex_sia', dense=True)
#    param_recon_3d_vertex(wpdc_param_62d_path, img_names_list, method = 'vertex_wpdc', dense=True)
#    param_recon_3d_vertex(param_fp_gt, img_names_list, method = 'vertex_gt', dense=True)

    # reconstruct 3D dense face shape, do not include pose
    param_recon_3d_vertex_shp(sia_param_62d_path, img_names_list, method = 'vertex_sia', dense=True)
    param_recon_3d_vertex_shp(wpdc_param_62d_path, img_names_list, method = 'vertex_wpdc', dense=True)
    param_recon_3d_vertex_shp(param_fp_gt, img_names_list, method = 'vertex_gt', dense=True)

    #analysis pose param
#    param_fp_gt='./train.configs/param_all_norm_val.pkl' 
#    sia_param_62d_path = 'result/sia_param_62d.npy'
#    wpdc_param_62d_path = 'result/wpdc_param_62d.npy'
#    gt_param = _load(param_fp_gt)
#    sia_param = _load(sia_param_62d_path)
#    wpdc_param = _load(wpdc_param_62d_path)
#    
#    dis_sia = nan_pose_diff(gt_param, sia_param)
#    dis_wpdc = nan_pose_diff(gt_param, wpdc_param)
#    cmp_dis = dis_sia-dis_wpdc
#    cmp_dis[cmp_dis<0] = 0
#    cmp_dis[cmp_dis>0] = 1
#    num = np.sum(cmp_dis)
    
    




































