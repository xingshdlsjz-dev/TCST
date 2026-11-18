import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        **kwargs,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Linear(output_size, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        pooled = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)
