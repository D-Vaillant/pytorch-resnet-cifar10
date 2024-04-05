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
    """
    """
    expansion = 1  # Bottleneck uses a different expansion. Not included here.

    def __init__(self, input_channels, output_channels, stride=1,
                 act_fn=None):
        super(ResidualBlock, self).__init__()
        self.act_fn = nn.ReLU() if act_fn is None else act_fn

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.convnet = nn.Sequential(self.conv1, self.bn1, self.act_fn,
                                     self.conv2, self.bn2)

        # The logic here is actually backwards:
        # if our stride is 1 and input_channels == output_channels, we use identity.
        # Otherwise we use Conv2d with k = 1 and the input stride.
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            # Adjusting if the dimensionality changes after the conv networks.
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        out = self.convnet(x) + self.shortcut(x)
        out = self.act_fn(out)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, Block, num_blocks: int,
                 input_channels: int, output_channels: int,
                 stride=1, act_fn=None):
        super(ResidualLayer, self).__init__()
        self.act_fn = nn.ReLU() if act_fn is None else act_fn

        layers = []
        layers.append(Block(input_channels, output_channels,
                            stride=stride, act_fn=act_fn))
        # Subsequent blocks have stride=1.
        for _ in range(1, num_blocks):
            layers.append(Block(output_channels, output_channels,
                                act_fn=act_fn))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResNet(nn.Module):
    """
    This is a ResNet. It is composed of a TRUNK and LAYERS.
    Each LAYER is composed of several BLOCKS.
    """
    resnet_act_fn = nn.ReLU()
    block_act_fn = None

    def __init__(self, block, num_blocks, num_classes=10,
                 image_channels=3, channels=64):
        super(ResNet, self).__init__()
        if self.block_act_fn is None:
            self.block_act_fn = nn.ReLU()

        # Note: Different block structures aren't supported just yet.
        # They can be hacked in by just multiplying the linear output dimension by 4.
        assert(len(num_blocks) == 4)
        self.channels = channels

        # k = 3, p = 1
        self.conv1 = nn.Conv2d(image_channels, self.channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels)
        layer_list = []

        for i, block_count in enumerate(num_blocks):
            # The first layer is 32x32, same as the input.
            # Then subsequent layers halve it: 16x16, 8x8, 4x4.
            if i == 0:
                block_stride = 1
                output_channels = self.channels
            else:
                block_stride = 2
                output_channels = 2 * self.channels

            layer_list.append(ResidualLayer(
                ResidualBlock, block_count, self.channels, output_channels,
                stride=block_stride, act_fn=self.block_act_fn
            ))
            self.channels = output_channels

        self.layers = nn.Sequential(*layer_list)
        # In 3-blocks, this should be 4 * self.channels.
        self.linear = nn.Linear(self.channels, num_classes)

    @property
    def trunk(self):
        return nn.Sequential(self.conv1, self.bn1, self.resnet_act_fn)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Trunk is the first conv layer, batch, and activation function.
        out = self.trunk(x)
        out = self.layers(out)
        # Hyperparameter: Pooling strategy.
        # out = F.max_pool2d(out, 4)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ELUResNet(ResNet):
    resnet_act_fn = nn.ELU()
    block_act_fn = nn.ELU()


class FatResNet(ResNet):
    """ResNet that uses a 5x5 kernel at the beginning."""
    def __init__(self, *args, image_channels=3, channels=64, **kwargs):
        super(FatResNet, self).__init__(*args,
                                        image_channels=image_channels,
                                        channels=channels,
                                        **kwargs)
        self.conv1 = nn.Conv2d(image_channels, channels,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
