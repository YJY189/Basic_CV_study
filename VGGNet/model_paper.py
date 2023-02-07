import torch
import torch.nn as nn

# convolution layer 2개인 block
def conv_2_block(in_dim, out_dim):
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    return net

# convolution layer 3개인 block
def conv_3_block(in_dim, out_dim):
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    return net

# convolution layer 4개인 block
def conv_4_block(in_dim, out_dim):
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    return net

class VGGNet16(nn.Module):
    def __init__(self, dataset='ImageNet', num_classes=1000):
        super().__init__()
        self.dataset = dataset
        self.net = nn.Sequential(
            # (3, 224, 224) or (3,32, 32)
            conv_2_block(3, 64),
            # (64, 112, 112) or (64, 16, 16)
            conv_2_block(64, 128),
            # (128, 56, 56) or (128, 8, 8)
            conv_3_block(128, 256),
            # (256, 28, 28) or (256, 4, 4)
            conv_3_block(256, 512),
            # (512, 14, 14) or (512, 2, 2)
            conv_3_block(512, 512),
            # (512, 7, 7) or (512, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        if dataset == 'cifar10':
            self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*1*1, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.net(x)
        if self.dataset == 'cifar10':
            x = x.view(-1, 512*1*1)
        else:
            x = x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x
            

class VGGNet19(nn.Module):
    def __init__(self, dataset='ImageNet', num_classes=1000):
        super().__init__()
        self.dataset = dataset
        self.net = nn.Sequential(
            # (3, 224, 224) or (3, 32, 32)
            conv_2_block(3, 64),
            # (64, 112, 112) or (64, 16, 16)
            conv_2_block(64, 128),
            # (128, 56, 56) or (128, 8, 8)
            conv_4_block(128, 256),
            # (256, 28, 28) or (256, 4, 4)
            conv_4_block(256, 512),
            # (512, 14, 14) or (512, 2, 2)
            conv_4_block(512, 512),
            # (512, 7, 7) or (512, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        if dataset == 'cifar10':
            self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*1*1, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.net(x)
        if dataset == 'cifar10':
            x = x.view(-1, 512*1*1)
        else:
            x = x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x