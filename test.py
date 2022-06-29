import argparse
import torch
from torchvision import transforms
import os
import opt
from torch.utils import data
import torch.nn as nn
import h5py
import numpy as np
from wind_dataset import WindDataset
from net import PConvUNet
from util.io import load_ckpt
from util.image import UnMinMaxUnNormalization

torch.multiprocessing.set_sharing_strategy('file_system')




def test_net(model, test_dataloader, device, args,
             save_dir=None, vis=False, save_tensor=False, no_compute_loss=True):
    model.eval()
    unminmax_unnormalization = UnMinMaxUnNormalization(device)
    with torch.no_grad():
        output_list = []
        gt_list = []
        mask_list = []
        for i, (image, mask, transform_gt, gt) in enumerate(test_dataloader):
            output, _ = model(image.to(device), mask.to(device))

            # if vis:
            #     output = output.to(torch.device('cpu'))
            #     output_comp = mask * image + (1 - mask) * output
            #     grid = make_grid(
            #         torch.cat((unnormalize(image), mask, unnormalize(output),
            #                    unnormalize(output_comp), unnormalize(gt)), dim=0))
            #     save_image(grid, os.path.join(save_dir, str(i) + '.png'))
            # gt = unnormalization(gt.to(device))
            output = unminmax_unnormalization(output)
            output_list.append(output.to('cpu'))
            gt_list.append(gt)

            mask_list.append(mask)
            if i % 100 == 0:
                print('processing {}'.format(i))

        output_tensor = torch.cat(output_list)
        gt_tensor = torch.cat(gt_list)
        mask_tensor = torch.cat(mask_list)
        output_np = output_tensor.numpy()
        gt_np = gt_tensor.numpy()
        mask_np = mask_tensor.numpy()

        if not no_compute_loss:
            # TODO: to modify your own evaluation metrics
            l1_loss_in_mask = compute_loss(output_tensor, gt_tensor, mask_tensor)
            print('l1_loss_in_mask', l1_loss_in_mask)
            correlation_all, rmse_all = compute_all(output_np, gt_np)
            print('Mean_series_correlation: {}; Mean_series_rmse: {}'.format(correlation_all, rmse_all))
            corr_grid, rmse_grid = compute_eachgrid(output_np, gt_np)
            print('Mean_grid_correlation: {}; Mean_grid_rmse: {}'.format(corr_grid, rmse_grid))
        if save_tensor:

            with h5py.File(os.path.join(save_dir, '{}_result.h5'.format(args.data_name.split('.')[0])), 'w') as f:
                f.create_dataset('output', data=output_np)
                f.create_dataset('gt', data=gt_np)
                f.create_dataset('mask', data=mask_np)

    model.train()

def compute_loss(output, gt, mask):
    l1 = nn.L1Loss(reduction='sum')
    # In mask, the value equal to 1 is retained and can be seen by the model
    # So, in evaluation phase, we should compute the unmask area.
    unmask = 1 - mask
    # Compute the loss of the unmask area.
    l1_loss = l1(output * unmask, gt * unmask)
    mask_count = mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3] - mask.count_nonzero()
    return l1_loss / mask_count
    # sum_l1_loss += l1_loss
    # sum_mask_count += mask_count


# Calculate the correlation and root mean square error of the total time series
def compute_all(output, gt):
    n = gt.shape[0]  # image number
    x = gt.reshape([n, -1]).mean(axis=1)
    y = output.reshape([n, -1]).mean(axis=1)
    correlation_all = np.corrcoef(x, y)[0, 1]
    rmse_all = np.sqrt(np.mean(np.power((x - y), 2)))
    return correlation_all, rmse_all


# Calculate the average value of correlation and root mean square error of time series on each grid
def compute_eachgrid(output, gt):
    output_t = output.transpose(2,3,0,1).reshape(output.shape[2]*output.shape[3], -1)
    gt_t = gt.transpose(2,3,0,1).reshape(gt.shape[2]*gt.shape[3], -1)
    x = gt_t.mean(axis=1)
    y = output_t.mean(axis=1)
    correlation_grid_m = np.corrcoef(x, y)[0,1]
    rmse_grid_m = np.sqrt(np.mean(np.power((x - y), 2)))
    return correlation_grid_m, rmse_grid_m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--root', type=str, default='/data/liuhaofeng/Dataset/win_speed/72x72')
    # parser.add_argument('--mask_root', type=str, default='/data/liuhaofeng/Dataset/win_speed/72x72/test')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--phrase', type=str, default='test')
    parser.add_argument('--data_name', type=str, default='test_data.h5')
    parser.add_argument('--mask_name', type=str, default='test_mask.h5')
    parser.add_argument('--model_name', type=str, default='win_speed2')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--iter', type=str, default='300000')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_threads', type=int, default=4)
    parser.add_argument('--save_tensor', action='store_true')
    parser.add_argument('--no_compute_loss', action='store_true')
    parser.add_argument('--snapshot', type=str, default='300000')
    parser.add_argument('--image_size', type=int, default=72)
    parser.add_argument('--random_mask', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')


    args = parser.parse_args()
    snapshot = os.path.join('snapshots', args.model_name, 'ckpt', args.iter + '.pth')
    save_dir = os.path.join(args.result_dir, args.model_name)
    # print(save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if args.use_cpu:
        torch.backends.cudnn.benchmark = False
        device = torch.device('cpu')
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda:{}'.format(args.gpu_id))

    size = (args.image_size, args.image_size)
    img_transform = transforms.Compose(
        [transforms.Resize(size=size),
         transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    mask_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor()])

    dataset = WindDataset(args.root, args.root, img_transform, mask_transform, args.phrase, args.data_name,
                          args.mask_name, random_mask=args.random_mask)

    dataloader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads)
    model = PConvUNet().to(device)
    load_ckpt(snapshot, [('model', model)])

    test_net(model, dataloader, device, args, save_dir=save_dir,
             save_tensor=args.save_tensor, no_compute_loss=args.no_compute_loss)
