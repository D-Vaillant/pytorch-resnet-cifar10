'''ResNet in PyTorch.

Original source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion = 1  # Bottleneck uses a different expansion. Not included here.
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != self.expansion*output_channels:
            # Adjusting if the dimensionality changes after the conv networks.
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, self.expansion*output_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*output_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.input_channels = 64
        self.channels = 64

        self.conv1 = nn.Conv2d(3, self.input_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_channels)
        layer_list = []

        for i, block_param in enumerate(num_blocks):
            stride = 1 if i == 0 else 2
            layer_list.append(self._make_layer(block, self.input_channels*(2**i), block_param, stride=stride))
        self.layers = nn.Sequential(*layer_list)
        ll = len(self.layers)
        self.linear = nn.Linear(block.expansion*self.input_channels*(2**(ll+1)), num_classes)

    def _make_layer(self, block, output_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.channels, output_channels, stride))
        self.channels = output_channels * block.expansion

        # Subsequent blocks have stride=1.
        for _ in range(1, num_blocks):
            layers.append(block(self.channels, output_channels))
            self.channels = output_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
