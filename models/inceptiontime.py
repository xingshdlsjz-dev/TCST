import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, bottleneck=32, out_channels=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(bottleneck, out_channels, kernel_size=ks, padding=ks // 2, bias=False)
                for ks in [9, 19, 39]
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.bottleneck(x)
        outs = [branch(z) for branch in self.branches]
        outs.append(self.pool_branch(x))
        return self.act(self.bn(torch.cat(outs, dim=1)))


class InceptionTime(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, layers=3, dropout=0.2, **kwargs):
        super().__init__()
        blocks = []
        in_channels = input_size
        for _ in range(layers):
            blocks.append(InceptionBlock(in_channels, bottleneck=hidden_size // 4, out_channels=hidden_size // 4))
            in_channels = hidden_size
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(self.features(x))
