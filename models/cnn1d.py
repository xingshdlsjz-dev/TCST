import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, dropout=0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.classifier(x)
