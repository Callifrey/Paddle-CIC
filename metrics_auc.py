import argparse
import os
from skimage import io, color
import numpy as np
from utils import check_file, get_paths
from paddle.vision import transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', type=str, default='/media/gallifrey/DJW/Dataset/Imagenet/val',
                    help='path for saving trained models')
parser.add_argument('--results_dir', type=str, default='./result', help='path for generated images')
parser.add_argument('--save_path', type=str, default='./metric/metric_results', help='path for save metric results')
args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

real_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
])

threshold_table = np.zeros(200)
real_paths = get_paths(args.imgs_dir)
fake_paths = get_paths(args.results_dir)
if len(real_paths) != len(fake_paths):
    raise (RuntimeError("the number of images for two dirs should be equal "))
for idx, (real_path, fake_path) in enumerate(zip(real_paths, fake_paths)):
    if not check_file(real_path, fake_path):
        raise (RuntimeError("image files are not match"))
    real_img = io.imread(real_path)
    real_img = real_transform(real_img)
    fake_img = io.imread(fake_path)
    real_ab = color.rgb2lab(real_img)[:, :, 1:3]
    fake_ab = color.rgb2lab(fake_img)[:, :, 1:3]
    ab_err = np.square(real_ab - fake_ab)
    ab_dist = np.sqrt(np.sum(ab_err, axis=2))
    for line in ab_dist:
        for item in line:
            threshold_table[int(item) + 1] += 1
    if idx % 10 == 0:
        print('Done step {}'.format(idx))

# calculate suffix sum of table
threshold_sum = threshold_table[:151]
for i in range(1, len(threshold_sum)):
    threshold_sum[i] += threshold_sum[i-1]

# write files
print('***** log results ******')
with open(os.path.join(args.save_path, 'result_log.txt'), 'w') as f:
    for idx, count in enumerate(threshold_sum):
        log = '{} {}\n'.format(idx, count)
        f.write(log)

# calculate AuC
for i in range(len(threshold_sum)):
    # get probability
    threshold_sum[i] /= 256*256*len(real_paths)
area = 0.0
for i in range(0, 150):
    area += (threshold_sum[i] + threshold_sum[i+1]) * 0.5
print('AuC:{}'.format(area / 150.0))
with open(os.path.join(args.save_path, 'auc_recond.txt'), 'w') as f:
        f.write('AuC:{}'.format(area / 150.0))

# plot curve
plt.figure()
x = [i for i in range(151)]
y = threshold_sum[:151]
plt.plot(x, y, marker='*', color='r')
plt.legend()
plt.xlabel('Euclidean distance in ab space')
plt.ylabel('Fraction of pixels within threshold')
plt.savefig(os.path.join(args.save_path,'Raw_AuC.png'))
plt.show()


