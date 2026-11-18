import torch
from torch import nn


class DeepConvLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        conv_channels=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        **kwargs,
    ):
        super().__init__()
        blocks = []
        in_channels = input_size
        for _ in range(4):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = conv_channels
        self.conv = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])
