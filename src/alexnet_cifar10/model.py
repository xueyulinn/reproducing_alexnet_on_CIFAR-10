import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, act, num_class):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn_linear_stack = nn.Sequential(
            # conv1: (227,227,3)
            nn.Conv2d(in_channels=3, out_channels=96,
                      kernel_size=11, stride=4),
            act,
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2: (55,55,96)
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=5, padding=2),
            act,
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
            # conv4
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3),
            # conv5
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3),
            act(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # fc6
            nn.Linear(in_features=9216, out_features=4096),
            act,
            nn.Dropout(),
            # fc7
            nn.Linear(in_features=4096, out_features=4096),
            act,
            nn.Dropout(),
            # fc8
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.cnn_linear_stack(x)
        return logits
