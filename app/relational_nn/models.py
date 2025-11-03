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
            # Graph-Transformer style: use TransformerConv on all relations; infer edge_dim per relation
            edge_dims: Dict[tuple[str, str, str], int] = {}
            for edge_type in data.edge_types:
                et_attr = getattr(data[edge_type], 'edge_attr', None)
                if et_attr is not None and isinstance(et_attr, torch.Tensor) and et_attr.dim() == 2:
                    edge_dims[edge_type] = int(et_attr.size(1))
                else:
                    edge_dims[edge_type] = 1

            conv_map = {
                edge_type: TransformerConv((-1, -1), self.hidden_dim, heads=4, concat=False, edge_dim=edge_dims[edge_type])
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
                    dim = 1 if edge_type not in edge_dims else edge_dims[edge_type]
                    edge_attr_dict[edge_type] = torch.zeros((E, dim), dtype=torch.float32, device=h_dict[dst].device)

            updated = conv(h_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            # Carry forward representations for node types that are not destinations
            for ntype, x in h_dict.items():
                if ntype not in updated:
                    updated[ntype] = x
            h_dict = {k: self.dropout(torch.relu(v)) for k, v in updated.items()}

        logits_end = self.head_end(h_dict["End_Action"])
        return {"End_Action": logits_end}



class SingleActionRelGraph(nn.Module):
    """
    Hetero GNN for single-node Action graphs.
    Nodes: Action, Player, Team. Predict next Action type for each Action.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int,
        out_dim: int,
        num_action_types: int,
        num_action_bodies: int,
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
        self.action_type_emb = nn.Embedding(num_action_types, hidden_dim)
        self.action_body_emb = nn.Embedding(num_action_bodies, hidden_dim)

        # Optional node ID embeddings for Player/Team
        self.player_id_emb = nn.Embedding(num_player_nodes, hidden_dim)
        self.team_id_emb = nn.Embedding(num_team_nodes, hidden_dim)

        # Numeric projection for Action: [period_id, time_seconds, start_x, start_y, end_x, end_y, is_home_team]
        self.lin_action = nn.Linear(7, hidden_dim)

        self.num_layers = num_layers
        self.head_action = nn.Linear(hidden_dim, out_dim)

    def _build_x(self, x_dict: Dict[str, Optional[torch.Tensor]], data: dict) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        xa = []
        if x_dict.get("Action") is not None:
            xa.append(self.lin_action(x_dict["Action"]))
        xa.append(self.action_type_emb(data["Action"].type_idx))
        xa.append(self.action_body_emb(data["Action"].body_idx))
        out["Action"] = self.dropout(sum(xa))

        out["Player"] = self.player_id_emb(torch.arange(self.player_id_emb.num_embeddings, device=out["Action"].device))
        out["Team"] = self.team_id_emb(torch.arange(self.team_id_emb.num_embeddings, device=out["Action"].device))
        return out

    def forward(self, data) -> Dict[str, torch.Tensor]:
        x_dict = {k: v for k, v in data.x_dict.items()}
        h_dict = self._build_x(x_dict, data)

        for _ in range(self.num_layers):
            # Infer per-relation edge_dim and build TransformerConv map
            edge_dims: Dict[tuple[str, str, str], int] = {}
            for edge_type in data.edge_types:
                et_attr = getattr(data[edge_type], 'edge_attr', None)
                if et_attr is not None and isinstance(et_attr, torch.Tensor) and et_attr.dim() == 2:
                    edge_dims[edge_type] = int(et_attr.size(1))
                else:
                    edge_dims[edge_type] = 1

            conv_map = {
                edge_type: TransformerConv((-1, -1), self.hidden_dim, heads=4, concat=False, edge_dim=edge_dims[edge_type])
                for edge_type in data.edge_types
            }
            conv = HeteroConv(conv_map, aggr="sum").to(h_dict["Action"].device)

            edge_attr_dict = {}
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                    edge_attr_dict[edge_type] = data[edge_type].edge_attr
            for edge_type in data.edge_types:
                if edge_type not in edge_attr_dict:
                    dst = edge_type[2]
                    E = data[edge_type].edge_index.size(1)
                    dim = 1 if edge_type not in edge_dims else edge_dims[edge_type]
                    edge_attr_dict[edge_type] = torch.zeros((E, dim), dtype=torch.float32, device=h_dict[dst].device)

            updated = conv(h_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            for ntype, x in h_dict.items():
                if ntype not in updated:
                    updated[ntype] = x
            h_dict = {k: self.dropout(torch.relu(v)) for k, v in updated.items()}

        logits = self.head_action(h_dict["Action"])
        return {"Action": logits}

