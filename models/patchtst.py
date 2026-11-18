import torch
from torch import nn


class PatchTSTHAR(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        seq_len=128,
        patch_len=16,
        stride=8,
        d_model=128,
        heads=4,
        layers=2,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = 1 + max(0, (seq_len - patch_len) // stride)
        self.patch_proj = nn.Linear(input_size * patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
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
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().permute(0, 1, 3, 2).flatten(start_dim=2)
        z = self.patch_proj(patches) + self.pos[:, : patches.shape[1]]
        z = self.encoder(z)
        return self.head(z.mean(dim=1))
