import torch
import torch.nn as nn

class SmallAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        ### convolution layers
        self.net = nn.Sequential(
            # conv1
            # (3, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.ReLU(),
            # (64, 26, 26)
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            # conv2
            # (64, 26, 26)
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(),
            # (192, 26, 26)
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (192, 12, 12)
            #conv3
            # (192, 12, 12)
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # (256, 12, 12)
            #conv4
            # (256, 12, 12)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # (256, 12, 12)
            #conv5
            # (256, 12, 12)
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # (128, 12, 12)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (128, 5, 5)
        )
        #fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(128*5*5), out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=num_classes),
        )
        
    # conv2d의 bias & weight 초기화
    def init_bias_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)   # weight 초기화
                nn.init.constant_(layer.bias, 0)   # bias 초기화
        # conv 2, 4, 5는 bias 1로 초기화 
        nn.init.constant_(self.net[3].bias, 1)
        nn.init.constant_(self.net[9].bias, 1)
        nn.init.constant_(self.net[11].bias, 1)
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 128*5*5)
        return self.classifier(x)