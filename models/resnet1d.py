import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResNet1D(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, layers=4, dropout=0.2, **kwargs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size, dropout=dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(self.blocks(self.stem(x)))
