import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from wind_dataset import WindDataset
from util.io import load_ckpt
from util.io import save_ckpt
from test import test_net
import torch.optim.lr_scheduler as lr_scheduler
#mp.set_start_method('spawn')
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/dataset/win_speed/72x72/')
parser.add_argument('--mask_root', type=str, default='/data/win_speed/72x72/')
parser.add_argument('--model_name', type=str, default='win_speed_train_project1')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=300000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=5000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--image_size', type=int, default=72)
parser.add_argument('--resume', type=str)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--loss_valid', type=float, default=1)
parser.add_argument('--loss_hole', type=float, default=6)
parser.add_argument('--loss_tv', type=float, default=0.1)
parser.add_argument('--loss_prc', type=float, default=0.05)
parser.add_argument('--loss_style', type=float, default=20)
parser.add_argument('--random_mask', action='store_true')


args = parser.parse_args()

LAMBDA_DICT = {
    'valid': args.loss_valid, 'hole': args.loss_hole, 'tv': args.loss_tv, 'prc': args.loss_prc, 'style': args.loss_style}

# 初始化
save_dir = os.path.join('snapshots', args.model_name)
log_dir = os.path.join('logs', args.model_name)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:{}'.format(args.gpu_id))
size = (args.image_size, args.image_size)

if not os.path.exists(save_dir):
    os.makedirs('{:s}/images'.format(save_dir))
    os.makedirs('{:s}/ckpt'.format(save_dir))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

# NOTE: remember to use minmax and normalize, the minmax scaler is encoded in WindDataset
img_transform = transforms.Compose(
    [transforms.Resize(size=size),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

# NOTE: the minmax scaler is encoded in WindDataset
dataset_train = WindDataset(args.root, args.mask_root, img_transform, mask_transform, 'train', image_size=args.image_size, random_mask=args.random_mask)
dataset_eval = WindDataset(args.root, args.mask_root, img_transform, mask_transform, 'eval', image_size=args.image_size)
dataset_test = WindDataset(args.root, args.mask_root, img_transform, mask_transform, 'test', image_size=args.image_size)


train_dataloader = data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True,
    num_workers=args.n_threads)
eval_dataloader = data.DataLoader(
    dataset_eval, batch_size=args.batch_size, shuffle=False,
    num_workers=args.n_threads)
test_dataloader = data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=False,
    num_workers=args.n_threads)

print(len(dataset_train))
model = PConvUNet(input_channels=1).to(device)

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, [args.max_iter*0.7, args.max_iter*0.9], gamma=0.3, last_epoch=-1)

criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

iter = 0
break_mark = False
model.train()

while iter < args.max_iter:
    for image, mask, gt, _ in train_dataloader:
        image = image.to(device)
        mask = mask.to(device)
        gt = gt.to(device)

        output, _ = model(image, mask)
        loss_dict = criterion(image, mask, output, gt)

        loss = 0.0

        for key, coef in LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value
            if (iter + 1) % args.log_interval == 0:
                writer.add_scalar('loss_{:s}'.format(key), value.item(), iter + 1)
                print('loss_{:s}: {} '.format(key, value.item()), end="")
        if (iter + 1) % args.log_interval == 0:
            print('iter {}: '.format(iter+1), end="")
            print()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iter + 1) % args.save_model_interval == 0 or (iter + 1) == args.max_iter or (iter + 1) == 1000:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(save_dir, iter + 1),
                      [('model', model)], [('optimizer', optimizer)], iter + 1)

        if (iter + 1) % args.vis_interval == 0 or (iter+1) == 200:
            model.eval()
            evaluate(model, dataset_eval, device,
                     '{:s}/images/test_{:d}.jpg'.format(save_dir, iter + 1))
            model.train()

        # counter
        iter += 1
        if iter > args.max_iter:
            break
        # validation
        if (iter+1) % 10000 == 0:
            print('learning rate ', scheduler.get_lr())
            print('evaluating eval dataset')
            test_net(model, eval_dataloader, device, args, no_compute_loss=False)
            print('evaluating test dataset')
            test_net(model, test_dataloader, device, args, no_compute_loss=False)
writer.close()
