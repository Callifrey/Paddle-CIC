import argparse
from paddle.vision import transforms
from paddle.io import DataLoader
from paddle.vision.models import vgg
from dataset import ImageNetClassification
import numpy as np
import paddle.fluid as fluid
import os

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='./result', help='path for generated images')
parser.add_argument('--save_path', type=str, default='./metric/metric_results_224', help='path for save metric results')
args = parser.parse_args()

model = vgg.vgg16(pretrained=True)
model.eval()
print(model)

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

imagenet = ImageNetClassification(root=args.results_dir, transform=val_transform)
dataloader = DataLoader(dataset=imagenet, batch_size=10, shuffle=False)

accuracy = []
for i, data in enumerate(dataloader):
    img, label = data
    label = label.reshape([-1, 1])
    predict = model(img)
    acc = fluid.layers.accuracy(predict, label)
    accuracy.append(acc)
    if i % 10 == 0:
        print('Done step {}'.format(i))
print('Accuracy:{}'.format(np.mean(accuracy)))

# log accuracy
with open(os.path.join(args.save_path, 'acc_recond.txt'), 'w') as f:
    f.write('Accuracy:{}'.format(np.mean(accuracy)))
