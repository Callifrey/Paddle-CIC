import argparse
import os
import sys

import paddle
from paddle.io import DataLoader
from skimage import io
from paddle.vision import transforms
from resources.training_layers import decode
from dataset import ImageNet
from model import Color_model
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=Warning)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./model', help='path for saving trained models')
parser.add_argument('--load_size', type=int, default=256, help='size for loading images')
parser.add_argument('--crop_size', type=int, default=256, help='size for randomly cropping images')
parser.add_argument('--image_dir', type=str, default='/media/gallifrey/DJW/Dataset/Imagenet/val', help='directory for resized images')
parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')
parser.add_argument('--load_model', type=str, default='19', help='the resume checkpoint')
parser.add_argument('--result_path', type=str, default='./result', help='path for results save')
parser.add_argument('--batch_size', type=int, default=20, help='data size for mini-batch ')
parser.add_argument('--num_workers', type=int, default=6, help='threads for load data')
parser.add_argument('--max_samples', type=int, default=int(sys.maxsize), help='the max size for test')

args = parser.parse_args()
print(args)

test_transform = transforms.Compose([
    transforms.Resize(args.load_size),
    transforms.CenterCrop(args.crop_size),
])

if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
# log_writer = LogWriter(logdir=args.log_path)

# 加载数据集
test_set = ImageNet(args.image_dir, test_transform)
test_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Build the models
model = Color_model()
model_path = os.path.join(args.model_path, 'epoch_{}_model.pdparams'.format(args.load_model))
state_dict = paddle.load(model_path)
model.load_dict(state_dict)
model.eval()


# Testing model
print('*******Testing*********')
for i, (L, _, _, path) in enumerate(test_data_loader):
    if i * args.batch_size >= args.max_samples:
        break
    outputs = model(L)
    for l, ab_feature, p in zip(L, outputs, path):
        fake_image = decode(l, ab_feature)
        fake_image = (fake_image * 255.0).astype(np.uint8)
        p_split = p.split('/')
        dir_name, file_name = p_split[-2], p_split[-1].split('.')[0]
        save_dir = os.path.join(args.result_path, dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        io.imsave(os.path.join(save_dir, '{}.png'.format(file_name)), fake_image)

    print('Done {} batches'.format(i))
