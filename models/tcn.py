import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.net(x) + self.downsample(x))


class TCN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, layers=4, dropout=0.2, **kwargs):
        super().__init__()
        blocks = []
        in_channels = input_size
        for i in range(layers):
            blocks.append(TemporalBlock(in_channels, hidden_size, kernel_size=5, dilation=2**i, dropout=dropout))
            in_channels = hidden_size
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(self.tcn(x))
