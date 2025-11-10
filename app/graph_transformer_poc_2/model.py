from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn


class MultiheadSelfAttentionWithEdgeBias(nn.Module):
    """
    Multi-head self-attention where the attention logits are adjusted with an additive bias C:
      A = (Q K^T) / sqrt(d_k) + C
    C is derived from edge features:
      - If an edge exists (i -> j) with feature e_ij, then c_ij = W1^T e_ij
      - Otherwise, if a shortest path exists, c_ij = W1^T (sum of edge features along the path)
      - Else c_ij = 0
    We receive a dense tensor edge_feature_sums of shape [N, N, E] that already encodes per-pair
    summed edge features (direct edge or shortest path within a hop limit). W1 maps E -> 1.
    """

    def __init__(self, embed_dim: int, num_heads: int, edge_feat_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # W1: edge_feat_dim -> 1, produces c_ij scalar per pair
        self.edge_scalar = nn.Linear(edge_feat_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, edge_feature_sums: torch.Tensor) -> torch.Tensor:
        """
        x: [N, d]
        edge_feature_sums: [N, N, E]
        returns: [N, d]
        """
        N, d = x.size()
        h = self.num_heads
        dk = self.head_dim

        q = self.q_proj(x).view(N, h, dk).transpose(0, 1)  # [h, N, dk]
        k = self.k_proj(x).view(N, h, dk).transpose(0, 1)  # [h, N, dk]
        v = self.v_proj(x).view(N, h, dk).transpose(0, 1)  # [h, N, dk]

        attn_logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(dk)  # [h, N, N]

        # Compute C from edge features and add to every head (shared scalar bias)
        c = self.edge_scalar(edge_feature_sums).squeeze(-1)  # [N, N]
        attn_logits = attn_logits + c.unsqueeze(0)  # broadcast over heads

        attn = torch.softmax(attn_logits, dim=-1)  # [h, N, N]
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [h, N, dk]
        out = out.transpose(0, 1).contiguous().view(N, d)  # [N, d]
        out = self.out_proj(out)  # [N, d]
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, edge_feat_dim: int, mlp_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttentionWithEdgeBias(embed_dim, num_heads, edge_feat_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, edge_feature_sums: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.norm1(x), edge_feature_sums)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ActionGraphTransformer(nn.Module):
    """
    Action-only Graph Transformer:
    - Node features: token embedding only (Action type); no numeric projections
    - Positional encoding: Laplacian eigenvectors projected and added to token
    - Attention: additive edge/path bias as specified (using precomputed per-pair edge feature sums)
    """

    def __init__(
        self,
        num_action_types: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        pos_enc_dim: int = 8,
        edge_feat_dim: int = 5,
        dropout: float = 0.1,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(num_action_types, embed_dim)
        self.pos_proj = nn.Linear(pos_enc_dim, embed_dim, bias=False)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, edge_feat_dim, mlp_ratio=4, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim if out_dim is not None else num_action_types)

    def forward(
        self,
        type_idx: torch.Tensor,
        lap_pos_enc: torch.Tensor,
        edge_feature_sums: torch.Tensor,
    ) -> torch.Tensor:
        """
        type_idx: [N] LongTensor with action type ids
        lap_pos_enc: [N, P] Laplacian positional encodings
        edge_feature_sums: [N, N, E] per-pair summed edge features (direct or shortest path)
        """
        x = self.token_emb(type_idx) + self.pos_proj(lap_pos_enc)
        for layer in self.layers:
            x = layer(x, edge_feature_sums)
        x = self.norm(x)
        logits = self.head(x)
        return logits  # [N, out_dim]


