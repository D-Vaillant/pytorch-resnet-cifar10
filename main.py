import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset


cifar10_train = torchvision.datasets.CIFAR10('cache', train=True, download=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

