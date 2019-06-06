#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:58:15 2019

@author: luoyao
"""

from sia_extract_param import extract_3DMM
from siamese_utils import reconstruct_vertex
from io_utils import _dump, _load, mkdir
import os.path as osp
import scipy.io as sio
from pathlib import Path

def aflw2000(data_info):
    # ji suan testset zhong san wei can shu, ge shi wei np.array
    params = extract_3DMM(data_info)
    _dump(data_info['params_save_path'], params)   #jiang  2000 zhang image de dao 62 wei de 3d paramers cun chu xia lai


def gen_3d_vertex_sel(filelists, aflw2000_sel_image, params_path, save_path):        # dense 3d face rather than 68 keypoints

    sel = Path(aflw2000_sel_image).read_text().strip().split('\n')
    fns = open(filelists).read().strip().split('\n')
    params = _load(params_path)       #jia zai 2000 ge yang ben de 3d data

    for i in range(2000):
        fn = fns[i]
        if fn in sel:
            vertex = reconstruct_vertex(params[i], dense=True)    #dense 3d face
            wfp = osp.join(save_path, fn.replace('.jpg', '.mat'))
            print(wfp)
            sio.savemat(wfp, {'vertex': vertex})

def gen_3d_vertex(filelists, params_path, save_path):        # dense 3d face rather than 68 keypoints

    fns = open(filelists).read().strip().split('\n')
    params = _load(params_path)       #jia zai 2000 ge yang ben de 3d data

    for i in range(2000):
        fn = fns[i]
        vertex = reconstruct_vertex(params[i], dense=True)    #dense 3d face
        wfp = osp.join(save_path, fn.replace('.jpg', '.mat'))
        print(wfp)
        sio.savemat(wfp, {'vertex': vertex})



if __name__ == '__main__':
    # step1: extract params
    checkpoint_fp_sia = 'training_debug/logs/wpdc+shp+identification_4_V_lr01/_checkpoint_epoch_50.pth.tar'   # model weight value
    checkpoint_fp_wpdc = 'training_debug/logs/wpdc_alpha/_checkpoint_epoch_50.pth.tar'   # model weight value
    root='test.data/AFLW2000-3D_crop'
    filelists='test.data/AFLW2000-3D_crop.list'
    
    mkdir('res/')
    params_sia_save_path = 'res/params_aflw2000_sia.npy'
    params_wpdc_save_path = 'res/params_aflw2000_wpdc.npy'

    aflw2000_sel_image = 'test.data/aflw2000_image_name.txt'
    mkdir('res/AFLW-2000-3D_vertex-sia/')
    mkdir('res/AFLW-2000-3D_vertex-wpdc/')
    sia_3d_vertex_path = 'res/AFLW-2000-3D_vertex-sia/'
    wpdc_3d_vertex_path = 'res/AFLW-2000-3D_vertex-wpdc/'
    
    data_info_sia = {'checkpoint_fp':checkpoint_fp_sia,'root':root,'filelists_test':filelists, 'params_save_path':params_sia_save_path}
    data_info_wpdc = {'checkpoint_fp':checkpoint_fp_wpdc,'root':root,'filelists_test':filelists, 'params_save_path':params_wpdc_save_path}
    
#    aflw2000(data_info_sia)
#    aflw2000(data_info_wpdc)


#    gen_3d_vertex_sel(filelists, aflw2000_sel_image, params_sia_save_path, sia_3d_vertex_path)
#    gen_3d_vertex_sel(filelists, aflw2000_sel_image, params_wpdc_save_path, wpdc_3d_vertex_path)


    gen_3d_vertex(filelists, params_sia_save_path, sia_3d_vertex_path)
    gen_3d_vertex(filelists, params_wpdc_save_path, wpdc_3d_vertex_path)



