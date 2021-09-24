import os
import numpy as np
import paddle
from PIL import Image
from paddle.io import Dataset
from paddle.vision import transforms
from skimage.color import rgb2lab

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir, classes_idx=None):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    if classes_idx is not None:
        assert type(classes_idx) == tuple
        start, end = classes_idx
        classes = classes[start:end]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in class_to_idx:
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# 用于colorization的数据集
class ImageNet(Dataset):
    def __init__(self, root, transform=None, loader=pil_loader, classes_idx=None):
        super().__init__()
        self.classes_idx = classes_idx
        classes, class_to_idx = find_classes(root, self.classes_idx)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img_original = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img_original)
        img_resize = transforms.Resize(64)(img_original)
        img_original = np.asarray(img_original)
        img_lab = rgb2lab(img_resize)
        img_ab = img_lab[:, :, 1:3]
        img_ab = paddle.to_tensor(img_ab.transpose((2, 0, 1)))
        img_l = rgb2lab(img_original)[:, :, 0] - 50.
        img_l = paddle.to_tensor(img_l, dtype=paddle.float32)

        return img_l.unsqueeze(0), img_ab, label, path

    def __len__(self):
        return len(self.imgs)

# 用于vgg 分类的数据集
class ImageNetClassification(Dataset):
    def __init__(self, root, transform=None, loader=pil_loader, classes_idx=None):
        super().__init__()
        self.classes_idx = classes_idx
        classes, class_to_idx = find_classes(root, self.classes_idx)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    from paddle.io import DataLoader

    train_root = '/media/gallifrey/DJW/Dataset/Imagenet/train'
    val_root = '/media/gallifrey/DJW/Dataset/Imagenet/val'
    original_transform = transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
    ])
    datasets = ImageNet(train_root, original_transform)
    print(len(datasets))  # 1279591
    datasets = ImageNet(val_root, original_transform)
    print(len(datasets))  # 50000
    data = DataLoader(datasets)
    for d in data:
        o, ab, label, path = d
        print(o.shape)  # [1, 256, 256]
        print(ab.shape)  # [1,2,64,64]
        print(label)  # label for img
        print(path)  # path for img
        break
