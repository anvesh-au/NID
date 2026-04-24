"""Flow encoder: MLP → L2-normalized embedding, trained with supervised contrastive loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
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
