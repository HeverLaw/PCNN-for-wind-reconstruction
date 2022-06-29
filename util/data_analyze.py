# -*- coding: UTF-8 -*-
"""
@Function:
@File: data_analyze.py
@Date: 2021/7/6 17:29 
@Author: Hever
"""
import h5py
import numpy as np
import os
from PIL import Image

data_root = '/data/liuhaofeng/Dataset/win_speed/144x144'
data_path = os.path.join(data_root, 'raw_data.h5')

# -------------------read the data---------------------
h5_file = h5py.File(data_path)
hdata = h5_file['data']

# use numpy to get the max, min, mean, variance value
hdata_np = hdata[:]
max_value = hdata_np.max()
hdata_np /= max_value
mean_value = hdata_np.mean()
std_value = hdata_np.std()
print(mean_value, std_value)
