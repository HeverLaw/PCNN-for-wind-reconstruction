import random
import torch
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')
import h5py
import numpy as np
import opt
from glob import glob
from util.mask_generator import RandomMaskingGenerator
import torch.utils.data as data


def minmaxscaler(data):
    return (data - opt.MIN[0]) / opt.MAX[0]


class WindDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train', data_name='test_data.h5', mask_name='test_mask.h5', image_size=144, random_mask=False):
        super(WindDataset, self).__init__()
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        # use about 8M images in the challenge dataset
        if split == 'train':
            self.data_path = glob('{:s}/train_data*.h5'.format(img_root))
            self.mask_path = glob('{:s}/train_mask*.h5'.format(mask_root))
        elif split == 'test':
            self.data_path = glob('{:s}/test_data*.h5'.format(img_root))
            self.mask_path = glob('{:s}/test_mask*.h5'.format(mask_root))
        elif split == 'eval':
            self.data_path = glob('{:s}/eval_data*.h5'.format(img_root))
            self.mask_path = glob('{:s}/eval_mask*.h5'.format(mask_root))
        elif split == 'observation':
            self.data_path = glob('{:s}/observation_data*.h5'.format(img_root))
            self.mask_path = glob('{:s}/observation_mask*.h5'.format(mask_root))
        elif split == 'inference':
            self.data_path = glob('{:s}/{:s}'.format(img_root, data_name))
            self.mask_path = glob('{:s}/{:s}'.format(mask_root, mask_name))
        else:
            raise AssertionError("Do not support split {}".format(split))
        self.max_transform = opt.MAX[0]

        print('loading data {}, {}...'.format(split, self.data_path[0].split('/')[-1]))
        # 尝试加载数据到内存中，但占的内存太大，不能这么操作
        self.data_file = h5py.File('{:s}'.format(self.data_path[0]), 'r')
        self.data = self.data_file.get('data')
        if self.data == None:
            self.data = self.data_file.get('gt')

        # self.data_np = torch.from_numpy(self.data[:]).float()
        self.mask_file = h5py.File(self.mask_path[0])
        self.mask_data = self.mask_file.get('mask')
        # self.mask_np = torch.from_numpy(self.mask_data[:].astype(np.float32)).float()
        print('loading data has been finished!')
        self.random_mask = random_mask
        self.mask_generator = RandomMaskingGenerator(image_size, 0.90, 0.90)

    def __getitem__(self, index):
        gt_img = np.expand_dims(self.data[index], 0)
        transform_gt_img = minmaxscaler(gt_img)

        transform_gt_img = torch.from_numpy(transform_gt_img).float()
        # transform_gt_img = transform_gt_img.unsqueeze(0)
        if self.random_mask:

            m = torch.from_numpy(self.mask_generator().astype(np.float32))
            mask = m.unsqueeze(0)
        else:
            mask = torch.from_numpy(self.mask_data[index].astype(np.float32))
            mask = mask.unsqueeze(0)

        # gt [0,25], mask[0,1], input=gt*mask, 然后再做transform
        input = transform_gt_img * mask
        input = self.img_transform(input)
        transform_gt_img = self.img_transform(transform_gt_img)
        # input = torch.ones((1, 80, 144))
        # mask = torch.ones((1, 80, 144))
        # transform_gt_img = torch.ones((1, 80, 144))
        # gt_img = torch.ones((1, 80, 144))
        return input, mask, transform_gt_img, gt_img

    def __len__(self):
        return len(self.data)
        # h5_file = h5py.File('{:s}'.format(self.data_path[0]), 'r')
        # hdata = h5_file.get('data')
        # leng = len(hdata[:,1,1])
        # return leng


# class TestDataset(torch.utils.data.Dataset):
#     def __init__(self, img_root, mask_root, img_transform, mask_transform):
#         super(TestDataset, self).__init__()
#
#         self.img_transform = img_transform
#         self.mask_transform = mask_transform
#         # use about 8M images in the challenge dataset
#         self.data_path = glob('{:s}/test_data*.h5'.format(img_root))
#         self.mask_path = glob('{:s}/test_mask2*.h5'.format(mask_root))
#         self.max_transform = 25.540
#
#     def __getitem__(self, index):
#         h5_file = h5py.File('{:s}'.format(self.data_path[0]), 'r')
#         hdata = h5_file.get('data')
#
#         gt_img = hdata[index, :, :]
#         gt_img_tensor = minmaxscaler(gt_img, self.max_transform)
#         gt_img = torch.from_numpy(gt_img_tensor[:,:]).float()
#         # gt_img = torch.from_numpy(gt_img[:, :]).float()
#         gt_img = gt_img.unsqueeze(0)
#
#         mask_file = h5py.File(self.mask_path[0])
#         mask_data = mask_file.get('mask')
#         mask = torch.from_numpy(mask_data[index, :, :].astype(np.float32))
#         mask = mask.unsqueeze(0)
#         # gt [0,25], mask[0,1], input=gt*mask, 然后再做transform
#         input = gt_img * mask
#         input = self.img_transform(input)
#         gt_img = self.img_transform(gt_img)
#
#         return input, mask, gt_img
#
#     def __len__(self):
#         h5_file = h5py.File('{:s}'.format(self.data_path[0]), 'r')
#         hdata = h5_file.get('data')
#         leng = len(hdata[:, 1, 1])
#         return leng

