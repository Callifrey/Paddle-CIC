import numpy as np
from PIL import Image
from utils import get_paths
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_val', type=str, default='/media/gallifrey/DJW/Dataset/Imagenet/val', help='root for validation set')
args = parser.parse_args()

pathes = get_paths(args.root_val)
recond = []
count = 0
for fn in pathes:
    img = Image.open(fn)
    img = np.array(img)
    count += 1
    if len(img.shape) != 3:
        recond.append(fn)
    if count % 100 == 0:
        print('done step {}'.format(count))
print(len(recond))

with open('remove_invalid.sh', 'w') as f:
    for item in recond:
        f.write('rm ' + item + '\n')




