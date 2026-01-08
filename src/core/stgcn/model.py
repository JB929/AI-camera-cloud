# src/core/stgcn/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """Simple temporal convolution block"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (N, C, T, V)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class STGCN_Backbone(nn.Module):
    """Lightweight ST-GCN backbone"""
    def __init__(self, in_channels=3, num_points=17):
        super().__init__()
        self.t1 = TemporalConv(in_channels, 32)
        self.t2 = TemporalConv(32, 64)
        self.t3 = TemporalConv(64, 64)
        self.num_points = num_points

    def forward(self, x):
        # x: (N, T, V, C)
        # Rearrange → (N, C, T, V)
        x = x.permute(0, 3, 1, 2)

        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)

        # Collapse spatial dimension
        # output: (N, C, T)
        x = x.mean(dim=3)
        return x


class STGCN_LSTM(nn.Module):
    """ST-GCN + LSTM final classifier"""
    def __init__(self, num_classes=6):
        super().__init__()

        self.backbone = STGCN_Backbone(in_channels=3, num_points=17)

        # LSTM expects (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (N, T, 17, 3)
        x = self.backbone(x)  # → (N, 64, T)

        # Rearrange for LSTM → (N, T, 64)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        out = out[:, -1]  # last timestep
        out = self.fc(out)
        return out

