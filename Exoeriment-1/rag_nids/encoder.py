"""Flow encoder: GLU residual MLP → L2-normalized embedding.

Upgrade from vanilla MLP: Gated Linear Units with residual connections. Feature-wise
gating acts as a learned soft feature selector per block; skip connections let gradients
flow cleanly through depth. Inference cost is unchanged (~1–3ms CPU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUBlock(nn.Module):
    """LayerNorm → gated linear unit → residual."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        v, g = self.gate(h).chunk(2, dim=-1)
        return x + self.drop(v * torch.sigmoid(g))


class FlowEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64, hidden: int = 256,
                 num_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.Sequential(*[GLUBlock(hidden, dropout) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(self.blocks(self.proj(x)))
        return F.normalize(z, dim=-1)


class EncoderWithAuxHead(nn.Module):
    """Encoder + auxiliary CE head — CE stabilizes SupCon on heavily imbalanced tabular data."""
    def __init__(self, encoder: FlowEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.aux = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z, self.aux(z)
