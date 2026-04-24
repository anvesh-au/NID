"""Cross-attention classifier head: fuses query embedding with k retrieved (embedding, label) tokens."""
import torch
import torch.nn as nn


class CrossAttentionHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, n_heads: int = 4, ffn_dim: int = 256):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.neighbor_proj = nn.Linear(embed_dim * 2, embed_dim)  # [z_neighbor || label_embed] -> d

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            batch_first=True, dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, query_z: torch.Tensor, neighbor_z: torch.Tensor, neighbor_labels: torch.Tensor):
        """
        query_z:         (B, D)
        neighbor_z:      (B, K, D)
        neighbor_labels: (B, K) int
        """
        lab_emb = self.label_embed(neighbor_labels)                  # (B, K, D)
        mem = self.neighbor_proj(torch.cat([neighbor_z, lab_emb], -1))  # (B, K, D)
        tgt = query_z.unsqueeze(1)                                   # (B, 1, D)
        out = self.decoder(tgt, mem)                                 # (B, 1, D)
        return self.cls(out.squeeze(1))                              # (B, num_classes)
