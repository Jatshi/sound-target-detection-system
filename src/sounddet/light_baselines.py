from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchCNN(nn.Module):
    def __init__(self, num_channels: int = 64, num_classes: int = 5):
        super().__init__()
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_feat = self.temporal_branch(x).flatten(1)
        spatial_feat = self.spatial_branch(x.unsqueeze(1)).flatten(1)
        return self.fc_fusion(torch.cat([temporal_feat, spatial_feat], dim=1))


class GTCNN(nn.Module):
    def __init__(self, num_channels: int = 64, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.channel_interaction = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.channel_interaction(x)
        return self.classifier(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ks1: int, ks2: int, use_1x1conv: bool = False, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=ks1, padding=(ks1 - 1) // 2, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=ks2, padding=(ks2 - 1) // 2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return F.relu(y + x)


class PaulNet5(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(128)
        self.res1 = ResBlock(128, 128, 3, 1)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.res2 = ResBlock(128, 128, 3, 3)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.res3 = ResBlock(128, 128, 3, 3)
        self.res4 = ResBlock(128, 128, 3, 3)
        self.mp3 = nn.MaxPool2d(kernel_size=2)
        self.res5 = ResBlock(128, 256, 1, 1, use_1x1conv=True)
        self.res6 = ResBlock(256, 512, 1, 1, use_1x1conv=True)
        self.res7 = ResBlock(512, 512, 1, 1)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.gap = nn.AvgPool2d(kernel_size=15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.mp1(self.res1(y))
        y = self.mp2(self.res2(y))
        y = self.mp3(self.res4(self.res3(y)))
        y = self.res7(self.res6(self.res5(y)))
        y = self.gap(self.bn2(self.conv2(y)))
        return y.flatten(1)


class LightBaselineWrapper(nn.Module):
    def __init__(self, name: str, num_classes: int = 5):
        super().__init__()
        self.name = name.lower()
        if self.name == "dualbranch":
            self.net = DualBranchCNN(num_channels=64, num_classes=num_classes)
        elif self.name == "gtcnn":
            self.net = GTCNN(num_channels=64, num_classes=num_classes)
        elif self.name == "paulnet":
            self.net = PaulNet5(num_classes)
        else:
            raise ValueError(f"Unknown lightweight baseline: {name}")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.name in {"dualbranch", "gtcnn"}:
            return self.net(spec.squeeze(1))
        if self.name == "paulnet":
            return self.net(F.interpolate(spec, size=(128, 128), mode="bilinear", align_corners=False))
        raise ValueError(f"Unknown lightweight baseline: {self.name}")
