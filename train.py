import argparse
import os
import paddle
import paddle.nn as nn
from paddle.vision import transforms
from paddle.io import DataLoader, DistributedBatchSampler
from resources.training_layers import PriorBoostLayer, NNEncLayer, NonGrayMaskLayer
from dataset import ImageNet
from model import Color_model
from visualdl import LogWriter
import paddle.distributed as dist
import warnings
warnings.filterwarnings("ignore", category=Warning)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
# 基本设置
parser.add_argument('--model_path', type=str, default='./model/', help='path for saving trained models')
parser.add_argument('--load_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--image_dir', type=str, default='/media/gallifrey/DJW/Dataset/Imagenet/train', help='directory for resized images')
parser.add_argument('--loss_step', type=int, default=10, help='step size for printing loss info')
parser.add_argument('--save_latest', type=int, default=2000, help='step for save model under iteration')
parser.add_argument('--save_freq', type=int, default=1, help='frequency for save models under epoch')
parser.add_argument('--log_path', type=str, default='./log', help='path for log files')
parser.add_argument('--continue_train', type=bool, default=False, help='if continue training')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch start training')

# 模型参数
parser.add_argument('--gpus', type=int, default=4, help='number of GPU for parallel training')
parser.add_argument('--num_epochs', type=int, default=20, help='total epochs for training')
parser.add_argument('--batch_size', type=int, default=40, help='size for mini-batch')
parser.add_argument('--num_workers', type=int, default=4, help='number of thread for data loader')
parser.add_argument('--lr', type=float, default=3.16e-5, help='initial learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 param for adam optimizer')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2 param for adam optimizer')
parser.add_argument('--weight_decay', type=float, default=0.001, help='rate for model weight decay')

# 超参数
parser.add_argument('--rebalance', type=bool, default=True, help='use color re-balance or not')
parser.add_argument('--NN', type=int, default=5, help='the number of nearest for KNN')
parser.add_argument('--sigma', type=float, default=5.0, help='sigma for gaussian kernel')
parser.add_argument('--gamma', type=float, default=0.5, help='rate for mixture of uniform distribution and empirical distribution')

args = parser.parse_args()
print(args)

train_transform = transforms.Compose([
    transforms.Resize(args.load_size),
    transforms.CenterCrop(args.crop_size)
])

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)
if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

log_writer = LogWriter(logdir=args.log_path)

# 加载数据集
train_set = ImageNet(args.image_dir, train_transform)
sampler = DistributedBatchSampler(train_set, batch_size=args.batch_size//args.gpus, shuffle=True, drop_last=True)
data_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.num_workers)
dist.init_parallel_env()

# 定义上色模型和操作层
model = Color_model()
model = paddle.DataParallel(model)
model.train()
if args.continue_train:
    state_dict = paddle.load(os.path.join(args.model_path, 'epoch_{}_model.pdparams'.format(args.which_epoch)))
    model.load_dict(state_dict)

# 定义用于编码ab space及进行color rebalance的层
encode_layer = NNEncLayer(NN=args.NN, sigma=args.sigma)
boost_layer = PriorBoostLayer(gamma=args.gamma)
nongray_mask = NonGrayMaskLayer()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(reduction='none')
params = list(model.parameters())
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=20000, gamma=0.9)
optimizer = paddle.optimizer.Adam(learning_rate=scheduler, beta1=args.beta1, beta2=args.beta2, parameters=params, weight_decay=args.weight_decay)


# 训练模型
total_step = len(data_loader)
count = 0
for epoch in range(args.num_epochs):
    for i, (images, img_ab, _, _) in enumerate(data_loader):
        outputs = model(images).transpose([0, 2, 3, 1])
        encode, max_encode = encode_layer.forward(img_ab)
        targets = paddle.to_tensor(max_encode, dtype=paddle.int64)
        if args.rebalance:
            boost = paddle.to_tensor(boost_layer.forward(encode), dtype=paddle.float32)
            mask = paddle.to_tensor(nongray_mask.forward(img_ab), dtype=paddle.float32)
            boost_nongray = boost*mask
            loss = (criterion(outputs, targets)*(boost_nongray.squeeze(1))).mean()
        else:
            loss = criterion(outputs, targets).mean()
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        count = count + 1

        # 打印loss信息
        if i % args.loss_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch, args.num_epochs, i, total_step, loss.item()))
            log_writer.add_scalar('train_loss', value=loss.item(), step=count)
        # 按step保存latest模型
        if count % args.save_latest == 0:
            paddle.save(model.state_dict(), os.path.join(
                args.model_path, 'epoch_latest_model.pdparams'))
    # 保存epoch模型
    if epoch % args.save_freq == 0:
        paddle.save(model.state_dict(), os.path.join(
            args.model_path, 'epoch_{}_model.pdparams'.format(epoch)))
