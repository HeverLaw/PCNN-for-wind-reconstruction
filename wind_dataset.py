import torch
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
        # TODO: modify your own dataset name
        if split == 'train':
            self.data_path = glob('{:s}/train_data.h5'.format(img_root))
            self.mask_path = glob('{:s}/train_mask.h5'.format(mask_root))
        elif split == 'test':
            self.data_path = glob('{:s}/test_data.h5'.format(img_root))
            self.mask_path = glob('{:s}/test_mask*.h5'.format(mask_root))
        elif split == 'eval':
            self.data_path = glob('{:s}/eval_data.h5'.format(img_root))
            self.mask_path = glob('{:s}/eval_mask.h5'.format(mask_root))
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

        self.data_file = h5py.File('{:s}'.format(self.data_path[0]), 'r')
        self.data = self.data_file.get('data')
        if self.data == None:
            self.data = self.data_file.get('gt')

        self.mask_file = h5py.File(self.mask_path[0])
        self.mask_data = self.mask_file.get('mask')
        print('loading data has been finished!')

        # Only design for MAE experiment. Default to choose False
        self.random_mask = random_mask
        self.mask_generator = RandomMaskingGenerator(image_size, 0.90, 0.90)

    def __getitem__(self, index):
        # gt_img is for testing only
        gt_img = np.expand_dims(self.data[index], 0)

        # NOTE: minmax scaler is encoded in WindDataset
        # minmaxscaler uses the MAX and MIN value in setting, is the same as to_tensor
        # the difference is our input is not the image with max value 255, min value 0.
        # Instead, they're MAX and MIN we defined.
        transform_gt_img = minmaxscaler(gt_img)
        transform_gt_img = torch.from_numpy(transform_gt_img).float()

        # Processing the mask
        if self.random_mask:
            m = torch.from_numpy(self.mask_generator().astype(np.float32))
            mask = m.unsqueeze(0)
        else:
            mask = torch.from_numpy(self.mask_data[index].astype(np.float32))
            mask = mask.unsqueeze(0)

        # mask the image, input=gt*mask
        input = transform_gt_img * mask
        # use img_transform(apply z-score)
        input = self.img_transform(input)
        transform_gt_img = self.img_transform(transform_gt_img)

        return input, mask, transform_gt_img, gt_img

    def __len__(self):
        return len(self.data)



