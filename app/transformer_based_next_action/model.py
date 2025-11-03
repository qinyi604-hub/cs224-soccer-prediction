from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        return x + self.pe[:, :n, :]


class SequenceTransformer(nn.Module):
    def __init__(
        self,
        num_types: int,
        num_bodies: int,
        num_players: int,
        num_teams: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        ff_dim: int = 256,
        num_numeric: int = 8,
    ) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(num_types, d_model)
        self.body_emb = nn.Embedding(num_bodies, d_model)
        self.player_emb = nn.Embedding(num_players, d_model)
        self.team_emb = nn.Embedding(num_teams, d_model)
        self.num_proj = nn.Linear(num_numeric, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.delta_proj = nn.Linear(1, d_model)
        self.head = nn.Linear(d_model, num_types)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # batch tensors: [B, K]
        e = self.type_emb(batch["type"]) + self.body_emb(batch["body"]) + self.player_emb(batch["player"]) + self.team_emb(batch["team"]) + self.num_proj(batch["num"])  # [B, K, d]
        x = self.pos(e) + self.delta_proj(batch["num"][..., -1:].contiguous())
        h = self.encoder(x)
        last = h[:, -1, :]
        logits = self.head(last)
        return logits


class GraphWindowTransformer(nn.Module):
    def __init__(
        self,
        num_types: int,
        num_bodies: int,
        num_players: int,
        num_teams: int,
        d_model: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(num_types, d_model)
        self.body_emb = nn.Embedding(num_bodies, d_model)
        self.player_emb = nn.Embedding(num_players, d_model)
        self.team_emb = nn.Embedding(num_teams, d_model)
        self.num_proj = nn.Linear(8, d_model)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True) for _ in range(num_layers)])
        # For graph-style attention on chain we will use TransformerConv-like effect via explicit edge attention using PyG is heavier;
        # We approximate by a stack of TransformerEncoderLayers after adding positional enc, then add an edge-aware pass using MHA mask
        self.pos = PositionalEncoding(d_model)
        self.delta_proj = nn.Linear(1, d_model)
        self.head = nn.Linear(d_model, num_types)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # x: [B, K, d]
        x = self.type_emb(batch["type"]) + self.body_emb(batch["body"]) + self.player_emb(batch["player"]) + self.team_emb(batch["team"]) + self.num_proj(batch["num"])
        # Add delta_t encoding from the last numeric channel
        x = self.pos(x) + self.delta_proj(batch["num"][..., -1:].contiguous())
        h = x
        for layer in self.layers:
            h = layer(h)
        last = h[:, -1, :]
        return self.head(last)


