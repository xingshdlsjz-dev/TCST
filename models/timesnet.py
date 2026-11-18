import torch
from torch import nn


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=(k, 3), padding=(k // 2, 1), bias=False),
                    nn.BatchNorm2d(16),
                    nn.GELU(),
                )
                for k in [3, 5, 7]
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        z = x.unsqueeze(1)
        out = torch.cat([branch(z) for branch in self.branches], dim=1)
        return x + self.proj(out).squeeze(1)


class TimesNetHAR(nn.Module):
    def __init__(self, input_size, num_classes, seq_len=128, d_model=128, layers=3, dropout=0.2, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.Sequential(*[MultiScaleTemporalBlock(d_model, dropout) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        z = self.input_proj(x)
        z = self.blocks(z)
        return self.head(z.mean(dim=1))
