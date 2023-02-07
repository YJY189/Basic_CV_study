import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        ### convolution layers
        self.net = nn.Sequential(
            # conv1
            # (3, 227, 227)
            # 논문에는 224라고 나와 있지만 conv1 이후 55 따르지 않고 54 따라서 227로 해줌
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            # (96, 55, 55)
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (96, 27, 27)
            # conv2
            # (96, 27, 27)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            # (256, 27, 27)
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (256, 13, 13)
            #conv3
            # (256, 13, 13)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            # (384, 13, 13)
            #conv4
            # (384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            # (384, 13, 13)
            #conv5
            # (384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # (256, 13, 13)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (256, 6, 6)
        )
        #fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256*6*6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        
    # conv2d의 bias & weight 초기화
    def init_bias_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)   # weight 초기화
                nn.init.constant_(layer.bias, 0)   # bias 초기화
        # conv 2, 4, 5는 bias 1로 초기화 
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256*6*6)
        return self.classifier(x)