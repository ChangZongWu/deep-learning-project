import os

import torch
import fastai
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import accuracy
import os
from fastai.callback.schedule import lr_find
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=test_transform
    )
    classes = train_data.classes
    img , label = train_data[0]
    print(img.shape, classes[label])
    def show_example(img, label):
        plt.imshow(img.permute(1, 2, 0))
        plt.title(classes[label])
        plt.axis(False)
        plt.show()

    # show_example(img, label)
    batch_size = 256
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    def show_batch(dataloader):
        for image, label in dataloader:
            fig, ax = plt.subplots(figsize=(16, 16))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(image[:100], nrow=10).permute(1, 2, 0))
            break
        plt.show()
    # show_batch(train_dataloader)
    class simple_residial_block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            return self.relu2(out + x)

    def conv_2d(in_channel,out_channel, stride= 1, ks = 3):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=ks,
                         stride=stride,
                         padding=ks//2)
    def batch_norm_relu_conv(in_channel, out_channel):
        return nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            conv_2d(in_channel=in_channel,
                     out_channel=out_channel)
        )
    class ResidualBlock(nn.Module):
        def __init__(self, in_channel, out_channel, stride = 1):
            super().__init__()
            self.bn = nn.BatchNorm2d(in_channel)
            self.conv1 = conv_2d(in_channel=in_channel,
                                  out_channel=out_channel,
                                  stride=stride)
            self.conv2 = batch_norm_relu_conv(out_channel,out_channel)
            self.shortcut = lambda x: x
            if in_channel != out_channel:
                self.shortcut = conv_2d(in_channel=in_channel,
                                         out_channel=out_channel,
                                         stride=stride,
                                         ks=1)
        def forward(self ,x):
            x = F.relu(self.bn(x), inplace=True)
            r = self.shortcut(x)
            x = self.conv1(x)
            x = self.conv2(x)*0.2
            return x.add_(r)
    def make_group(N , in_channel,out_channel, stride):
        start = ResidualBlock(in_channel=in_channel,
                              out_channel=out_channel,
                              stride=stride)
        rest = [ResidualBlock(in_channel=out_channel,
                              out_channel=out_channel,
                              stride=1) for j in range(1, N)]
        return [start] + rest
    class WideResNet(nn.Module):
        def __init__(self, n_group, N, n_classes, k = 1, n_start = 16):
            super().__init__()
            layers = [conv_2d(3, n_start)]
            n_channels = [n_start]
            for i in range(n_group):
                n_channels.append(n_start * (2**i) * k)
                stride = 2 if i > 0 else 1
                layers += make_group(N, n_channels[i], n_channels[i+1], stride)

            layers += [nn.BatchNorm2d(n_channels[3]),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(n_channels[3], n_classes)]
            self.feature = nn.Sequential(*layers)
        def forward(self,x):
            return self.feature(x)
    def wrn_22():
        return WideResNet(n_group=3, N=3, n_classes=len(classes), k=6)

    model = wrn_22()
    data = DataLoaders(train_dataloader, test_dataloader)
    learn = Learner(data, model,loss_func=F.cross_entropy, metrics=[accuracy])
    learn.clip = 0.1
    learn.lr_find()
    #learn.recorder.plot_lr_find()
    learn.fit_one_cycle(10, 5e-3, wd=1e-4)


