import paddle
import paddle.nn as nn


class Color_model(nn.Layer):
    def __init__(self):
        super(Color_model, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),  # (H/2, W/2)
            nn.ReLU(),
            nn.BatchNorm2D(num_features=64),
            # conv2
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),  # (H/4,W/4)
            nn.ReLU(),
            nn.BatchNorm2D(num_features=128),
            # conv3
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # (H/8,W/8)
            nn.ReLU(),
            nn.BatchNorm2D(num_features=256),
            # conv4
            nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (H/8,H/8)
            nn.ReLU(),
            nn.BatchNorm2D(num_features=512),
            # conv5
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            # (H/8,W/8)
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            # ks=3, stride=1, pad = 2, dilation=2的扩张卷积不改变特征图尺寸
            nn.ReLU(),
            nn.BatchNorm2D(num_features=512),
            # conv6
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            # (H/8,W/8)
            nn.ReLU(),
            nn.BatchNorm2D(num_features=512),
            # conv7
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2D(num_features=512),
            # conv8
            nn.Conv2DTranspose(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            # (H/4,W/4)
            nn.ReLU(),
            # conv8_313
            nn.Conv2D(in_channels=256, out_channels=313, kernel_size=1, stride=1, dilation=1),
            # （H/4, W/4, 313)
        )

    def forward(self, gray_image):
        features = self.features(gray_image)
        return features


if __name__ == '__main__':
    img = paddle.rand([1, 1, 256, 256])
    model = Color_model()
    feature = model(img)
    print(feature.shape)  # [1, 313, 64, 64]
