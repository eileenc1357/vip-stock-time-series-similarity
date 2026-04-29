import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(self, seq_len=30, patch_len=5, d_model=64, num_heads=4):
        super().__init__()

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len

        self.patch_embed = nn.Linear(patch_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Linear(d_model * self.num_patches, 1)

    def forward(self, x):
        B, L, C = x.shape
        x = x.squeeze(-1)

        x = x[:, :self.num_patches * self.patch_len]
        x = x.reshape(B, self.num_patches, self.patch_len)

        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x.reshape(B, -1)

        return self.head(x)