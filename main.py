'''Train CIFAR10 with PyTorch.
Source: https://raw.githubusercontent.com/kuangliu/pytorch-cifar/master/main.py'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms.v2 as transforms

import os
import argparse
import logging
import json

from model import *

USE_PROGRESS_BAR = True

if USE_PROGRESS_BAR:
    from utils import progress_bar

import datetime

def get_unique_identifier():
    current_datetime = datetime.datetime.now()
    unique_identifier = current_datetime.strftime("%Y%m%d_%H%M%S")
    return unique_identifier


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epochs', '-e', default=200, type=int, help='epochs to run. in addition to whatever the checkpoint')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--checkpoint', '-c', default=None,
                        help='checkpoint filename to load from. stored in checkpoint/.')
    return parser


class CIFAR10Trainer:
    def __init__(self, epochs: int,
                 learning_rate: float,
                 model: nn.Module,
                 model_args: dict,
                 checkpoint=None):
        self.epochs = epochs
        if torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.model_args = model_args
        self.run_name = get_unique_identifier()

        self.prepare_data()
        self.build_model(model, self.device, model_args, checkpoint=checkpoint)
        self.define_criterion_optimizer(learning_rate=learning_rate)

    def prepare_data(self):
        logging.info('==> Preparing data..')
        # Data augmentation scheme.
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=True, num_workers=2)
            
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=2)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def build_model(self, model, device, model_args, checkpoint=None):
        logging.info('==> Building model..')
        self.net = model(ResidualBlock, **model_args)
        if checkpoint is not None:  # We're passing in a checkpoint. Let's load from it.
            logging.info('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(f'./checkpoint/{checkpoint}.pth')
            # Fix for loading CUDA-trained nets on non-CUDA devices.
            checkpoint['net'] = {k.replace("module.", ""): v for k, v in checkpoint['net'].items()}
            try:
                assert model.__class__.__name__ == checkpoint['name'], 'Model names do not match.'
            except KeyError:
                pass  # Older checkpoint without a saved model name. It's fine.
            # self.block_info = checkpoint.get('layers')
            self.net = self.net.to(self.device)
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        if device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        self.net = self.net.to(device)

    def define_criterion_optimizer(self, learning_rate):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if USE_PROGRESS_BAR:
                progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if not USE_PROGRESS_BAR:    
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        self.train_losses.append(train_loss/(batch_idx+1))
        self.train_accuracies.append(100.*correct/total)

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if USE_PROGRESS_BAR:
                    progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if not USE_PROGRESS_BAR:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))      
        self.test_losses.append(test_loss/(batch_idx+1))
        self.test_accuracies.append(100.*correct/total)

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            logging.info('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'name': self.net.__class__.__name__,
                'model_args': self.model_args
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/ckpt_{self.run_name}.pth')
            self.best_acc = acc

    def export_results(self):
        results = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }
        
        if not os.path.isdir('results'):
            os.mkdir('results')
        
        with open(f'./results/training_results_{self.run_name}.json', 'w') as f:
            json.dump(results, f)
        
        print(f'Training results exported to ./results/training_results_{self.run_name}.json')

    def run(self):
        for epoch in range(self.start_epoch, self.start_epoch+self.epochs):
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()
        
        self.export_results()

class HeavilyAugmentedCIFAR10Trainer(CIFAR10Trainer):
    def prepare_data(self):
        logging.info('==> Preparing data..')
        # Data augmentation scheme.
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=True, num_workers=2)
            
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=2)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    model_args = {'num_blocks': [5, 4, 3], 'channels': 64}

    model = SiLUResNet(ResidualBlock, **model_args)
    print(f"Model parameter count: {model.parameter_count}")
    del model

    trainer = CIFAR10Trainer(model=SiLUResNet, model_args=model_args,
                             epochs=args.epochs, learning_rate=args.lr, checkpoint=args.checkpoint)
    trainer.run()
