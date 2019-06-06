#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 09:58:51 2019

@author: luoyao
"""

import os
import numpy as np
import torch
import pickle
import os.path as osp
from pathlib import Path
from collections import defaultdict
import random

def _get_suffix(filename):
    """ a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos+1:]


def _remove_suffix(filename):
    """xxxx_yyyy_ccc.jpg -> xxxx_yyyy_ccc"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[:pos]



def split_img_name(f_name):
    if len(f_name) == 0:
        raise Exception(f'sting is empty!')
    f_s = f_name.split('_')
    if len(f_s) == 0:
        raise Exception(f"split '_' error!")        
    key_name = f_s[2]+'_'+f_s[3]
    
    return key_name
 
    
def split_dataset_name(f_name):
    if len(f_name) == 0:
        raise Exception(f'sting is empty!')
    f_temp = _remove_suffix(f_name)
    f_s = f_temp.split('_')
    if len(f_s) == 0:
        raise Exception(f"split '_' error!")
    dataset_name = ''
    for f in f_s:
        if f.isalpha():
            dataset_name = dataset_name+'_'+f 
    pos = dataset_name.find('_')
    dataset_name = dataset_name[pos+1:]
    return dataset_name    


def split_digit_folder(f_name):
    if len(f_name) == 0:
        raise Exception(f'sting is empty!')
    f_temp = _remove_suffix(f_name)
    f_s = f_temp.split('_')
    if len(f_s) == 0:
        raise Exception(f"split '_' error!")
    pos = 0
    for f in f_s:
        if f.isalpha():
            pos = pos+1
    fir_folder_name = f_s[pos]
    
    return fir_folder_name


def num_of_digit(f_name):
    if len(f_name) == 0:
        raise Exception(f'sting is empty!')
    f_temp = _remove_suffix(f_name)
    f_s = f_temp.split('_')
    if len(f_s) == 0:
        raise Exception(f"split '_' error!")
    num=0
    for f in f_s:
        if f.isdigit():
            num = num+1
    return num


def split_digit_folder_num(f_name):
    if len(f_name) == 0:
        raise Exception(f'sting is empty!')
    f_temp = _remove_suffix(f_name)
    f_s = f_temp.split('_')
    if len(f_s) == 0:
        raise Exception(f"split '_' error!")
    pos = 0
    for f in f_s:
        if f.isalpha():
            pos = pos+1
    fir_folder_name = f_s[pos]+'_'+f_s[pos+1]
    
    return fir_folder_name


def get_keyname(num, f_name):
        
    if num == 4:
        dataset_name = split_dataset_name(f_name)
        pic_person_name = split_digit_folder_num(f_name)
    elif num == 3:
        dataset_name = split_dataset_name(f_name)
        pic_person_name = split_digit_folder(f_name)
    else:
        raise Exception(f'num is error!')
    key_name = dataset_name+'_'+pic_person_name
    
    return key_name


def label_original_filenames(label, ori_filenames):
    new_filenames = []
    for filename in ori_filenames:
        if num_of_digit(filename) == 4:          
            key_name = get_keyname(4, filename)
        elif num_of_digit(filename) == 3:
            key_name = get_keyname(3, filename)
        else:
            raise Exception(f'Nnknown style!!!')
            
        class_num = label[key_name]
        new_filename = f'{filename}\000{class_num}'
        new_filenames.append(new_filename)
    
    return new_filenames        


def verification(new_filenames, ori_filenames):
    
    right_num = 0
    error_img_name = []
    for i, filename in enumerate(new_filenames,0):
        f_temp = filename.split('\000')
        if f_temp[0] == ori_filenames[i]:
            right_num = right_num+1
        else:
            error_img_name.append(filename)
    return right_num, error_img_name

def extract_pic_path(dataset_path):
    pic_path_list = []
    all_info = os.walk(dataset_path)
    for folder_path, dir_, file_list in all_info:
        for filename in file_list:
            if filename.endswith('jpg') or filename.endswith('png'):
                filename = '/'+filename
                pic_path = folder_path + filename
                pic_path_list.append(pic_path)
    return pic_path_list

def path_list_gen_txt(path_list, txt_path):

    f = open(txt_path, 'w',encoding="utf-8")
    for filename in path_list:
        f.write('%s\n' % filename)
    f.close()

def create_label_dict(path):
    label_dict = defaultdict(list)
    names_list = Path(path).read_text().strip().split('\n')
    for f_name in names_list:
        f_s = f_name.split('\000')
        label_dict[int(f_s[1])].append(f_s[0])
    
    return label_dict

def make_pairs(label_dict):
    txt_list = []
    num_same = 0
    for i in range(6000):
        label_1 = random.choice( range(len(label_dict)) )
        order = random.choice(range(len(label_dict[label_1])))
        img1_name = label_dict[label_1].pop(order)
        is_same = random.randint(0,1)    
#        is_same = 1    #constrict the same people
        
        if is_same:
            order = random.choice(range(len(label_dict[label_1])))
            img2_name = label_dict[label_1].pop(order)
            num_same = num_same + 1
        else:
            while True:
                label_2 = random.choice( range(len(label_dict)) )
                if label_2 != label_1:
                    break
            order = random.choice(range(len(label_dict[label_2])))
            img2_name = label_dict[label_2].pop(order)
        
        line_info = img1_name + "\t" + img2_name +"\t"+ str(is_same)
        txt_list.append(line_info)
    return txt_list, num_same

def write_paris_txt(txt_list, txt_path):
    f = open(txt_path, 'w',encoding="utf-8")
    for line in txt_list:
        f.write('%s\n' % line)
    f.close()


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split()
            pairs.append(pair)
    return pairs

if __name__ == "__main__":
    
    filelists = './train_aug_120x120.list.val'
    filenames = Path(filelists).read_text().strip().split('\n')
    filenames_copy = filenames[:]
#
#    #generate person dict
    dic_keyname = {}
    dic = defaultdict(list)
    for filename in filenames_copy:
        if num_of_digit(filename) == 4:    
            img_name = get_keyname(4, filename)
            dic[img_name].append(filename)
        elif num_of_digit(filename) == 3:
            img_name_ = get_keyname(3, filename)
            dic[img_name_].append(filename)
        else:
            raise Exception(f'Nnknown style!!!')
#   #generate keyname dict
    class_num = 0
    for keyname in dic.keys():
        dic_keyname.setdefault(keyname, class_num)
        class_num = class_num + 1
#    
    for key, order in dic_keyname.items():
        if order == 60:  #28 for paper
            find = key
    
#    #keep original sort, and label it        
    new_filenames = label_original_filenames(dic_keyname, filenames)        
#     
#    
#    #verification
#    right_num, error_img_name = verification(new_filenames, filenames)       
#            
#    #write txt file
    label_path = './label_train_aug_120x120.txt'
    f = open(label_path, 'w',encoding="utf-8")
    for filename_ in new_filenames:
        f.write('%s\n' % filename_)
    f.close()

#    #verification
#    names_list = Path(label_path).read_text().strip().split('\n')
#    right_num_, error_img_name_ = verification(names_list, filenames)          
            
    
    ### extract lfw pic path, and make txt 
#    dataset_path = '/home/luoyao/Project_3d/compare-test/facenet-master/datasets/align_lfw'
#    txt_path = '/home/luoyao/Project_3d/compare-test/facenet-master/datasets/lfw.txt'
#    pic_path_list = extract_pic_path(dataset_path)
#    path_list_gen_txt(pic_path_list, txt_path)
    
    
    #make 6000 pairs person
#    filelists_val = "./label_train_aug_120x120.list.val"
#    prirs_path = "/home/luoyao/Project_3d/3D_face_solution/Siamese_network/recongnition/pairs_ddfa.txt"
#    label_dic = create_label_dict(filelists_val)
#    txt_list, num_same = make_pairs(label_dic)
#    write_paris_txt(txt_list, prirs_path)
#    
#    pairs = read_pairs(prirs_path)





















