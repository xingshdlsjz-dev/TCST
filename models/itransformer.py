import torch
from torch import nn


class ITransformerHAR(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        seq_len=128,
        d_model=128,
        heads=4,
        layers=2,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.channel_embedding = nn.Parameter(torch.zeros(1, input_size, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.value_embedding(x) + self.channel_embedding[:, : x.shape[1]]
        z = self.encoder(z)
        return self.head(z.mean(dim=1))
