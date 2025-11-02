from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, TransformerConv


class RelGraphSAGE(nn.Module):
    """
    Simple hetero relational GNN using HeteroConv with SAGEConv per relation.
    It consumes numeric node features and augments Start/End actions with learned
    embeddings for categorical fields (type/bodypart). Player/Team get learned
    node ID embeddings if their x is missing.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int,
        out_dim: int,
        num_start_types: int,
        num_start_bodies: int,
        num_end_types: int,
        num_end_bodies: int,
        num_player_nodes: int,
        num_team_nodes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Categorical embeddings
        self.start_type_emb = nn.Embedding(num_start_types, hidden_dim)
        self.start_body_emb = nn.Embedding(num_start_bodies, hidden_dim)
        self.end_type_emb = nn.Embedding(num_end_types, hidden_dim)
        self.end_body_emb = nn.Embedding(num_end_bodies, hidden_dim)

        # Optional node ID embeddings for Player/Team
        self.player_id_emb = nn.Embedding(num_player_nodes, hidden_dim)
        self.team_id_emb = nn.Embedding(num_team_nodes, hidden_dim)

        # Linear projections for numeric features per node type
        self.lin_start = nn.Linear(5, hidden_dim)
        self.lin_end = nn.Linear(5, hidden_dim)

        # Hetero layers
        self.num_layers = num_layers

        # Prediction head for End_Action nodes (predict next Start type)
        self.head_end = nn.Linear(hidden_dim, out_dim)

    def _build_x(self, x_dict: Dict[str, Optional[torch.Tensor]], data: dict) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        # Start_Action: numeric + cat embeddings
        xs = []
        if x_dict.get("Start_Action") is not None:
            xs.append(self.lin_start(x_dict["Start_Action"]))
        xs.append(self.start_type_emb(data["Start_Action"].type_idx))
        xs.append(self.start_body_emb(data["Start_Action"].body_idx))
        out["Start_Action"] = self.dropout(sum(xs))

        # End_Action: numeric + cat embeddings
        xe = []
        if x_dict.get("End_Action") is not None:
            xe.append(self.lin_end(x_dict["End_Action"]))
        xe.append(self.end_type_emb(data["End_Action"].type_idx))
        xe.append(self.end_body_emb(data["End_Action"].body_idx))
        out["End_Action"] = self.dropout(sum(xe))

        # Player/Team: learned ID embeddings
        out["Player"] = self.player_id_emb(torch.arange(self.player_id_emb.num_embeddings, device=out["End_Action"].device))
        out["Team"] = self.team_id_emb(torch.arange(self.team_id_emb.num_embeddings, device=out["End_Action"].device))
        return out

    def forward(self, data) -> Dict[str, torch.Tensor]:
        x_dict = {k: v for k, v in data.x_dict.items()}
        h_dict = self._build_x(x_dict, data)

        for _ in range(self.num_layers):
            # Graph-Transformer style: use TransformerConv on all relations with 1-d edge_attr
            conv_map = {
                edge_type: TransformerConv((-1, -1), self.hidden_dim, heads=1, edge_dim=1)
                for edge_type in data.edge_types
            }
            conv = HeteroConv(conv_map, aggr="sum").to(h_dict["End_Action"].device)

            edge_attr_dict = {}
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                    edge_attr_dict[edge_type] = data[edge_type].edge_attr
            # Ensure edge_attr exists for all relations (required by TransformerConv)
            for edge_type in data.edge_types:
                if edge_type not in edge_attr_dict:
                    dst = edge_type[2]
                    E = data[edge_type].edge_index.size(1)
                    edge_attr_dict[edge_type] = torch.zeros((E, 1), dtype=torch.float32, device=h_dict[dst].device)

            updated = conv(h_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            # Carry forward representations for node types that are not destinations
            for ntype, x in h_dict.items():
                if ntype not in updated:
                    updated[ntype] = x
            h_dict = {k: self.dropout(torch.relu(v)) for k, v in updated.items()}

        logits_end = self.head_end(h_dict["End_Action"])
        return {"End_Action": logits_end}


