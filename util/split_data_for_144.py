# -*- coding: UTF-8 -*-
"""
@Function: Divide the data into training set and verification set according to the ratio of 8:1
@File: create_data.py
@Date: 2021/7/4 16:45 
@Author: Hever
"""
import h5py
import numpy as np
import os
from PIL import Image

# set the global variable
np.random.seed(2021)
split_ratio = 8 / 9
# set the data root and the path to the mask
data_root = '/data/liuhaofeng/Dataset/win_speed/144x144_0820'
mask_path = '/data/liuhaofeng/Dataset/win_speed/144x144_0820/mask_template.h5'


def split_data_train_eval(data_path):
    """
    split the data to get training and evaluation dataset
    :param data_path: path to data
    """
    # -------------------read data-----------------------------
    h5_file = h5py.File(data_path)
    hdata = h5_file['data']

    # ------------------separate the data----------------------
    # Randomly select the training set and validation set with the ratio of 8:1
    shuffle_index = np.random.permutation(len(hdata))
    train_index = np.sort(shuffle_index[:int(len(hdata) * split_ratio)])
    eval_index = np.sort(shuffle_index[int(len(hdata) * split_ratio):])
    print(train_index)
    train_data = hdata[train_index]
    eval_data = hdata[eval_index]

    # --------------------save the data-----------------------
    with h5py.File(os.path.join(data_root, 'train_data.h5'), 'w') as f:
        f.create_dataset('data', data=train_data)

    with h5py.File(os.path.join(data_root, 'eval_data.h5'), 'w') as f:
        f.create_dataset('data', data=eval_data)


def split_mask_train_eval(data_path, mask_path):
    """
    split the mask data to get training and evaluation mask dataset
    :param data_path: path to data
    :param mask_path: path to mask
    :return:
    """
    # Each data corresponds to its unique mask
    np.random.seed(2022)

    # -------------------read data--------------------------
    h5_file = h5py.File(data_path)
    hdata = h5_file['data']

    # -------------------With put back sampling--------------
    # training dataset
    split_mask(mask_path, int(len(hdata) * split_ratio), 'train')
    # validation dataset
    split_mask(mask_path, int(len(hdata) * (1 - split_ratio)), 'eval')


def split_mask(mask_path, output_mask_num, phrase):
    h5_file = h5py.File(mask_path)
    h5_data = h5_file['mask']
    h5_data = h5_data[:]
    input_mask_num = len(h5_data)
    mask_index_list = np.random.choice(input_mask_num, output_mask_num, replace=True)
    mask_data = h5_data[mask_index_list]
    with h5py.File(os.path.join(data_root, '{}_mask.h5'.format(phrase)), 'w') as f:
        f.create_dataset('mask', data=mask_data)


if __name__ == '__main__':
    # Setp1: divide the data into training data and verification set data
    split_data_train_eval(os.path.join(data_root, 'raw_data.h5'))

    # Setp2: divide the mask into train and eval
    split_mask_train_eval(os.path.join(data_root, 'raw_data.h5'), mask_path)

    # h5_file = h5py.File('wind_speed/train_data.h5')
    # hdata = h5_file['train_data']
    # print()
    # h5_file = h5py.File('wind_speed/train_mask.h5')
    # hdata = h5_file['mask']
    # print()




