import torch
from torch import nn


class GRUFCN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, dropout=0.25, **kwargs):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fcn = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2 + 128, num_classes),
        )

    def forward(self, x):
        _, h = self.gru(x)
        gru_feat = torch.cat([h[-2], h[-1]], dim=1)
        fcn_feat = self.fcn(x.transpose(1, 2))
        return self.head(torch.cat([gru_feat, fcn_feat], dim=1))
