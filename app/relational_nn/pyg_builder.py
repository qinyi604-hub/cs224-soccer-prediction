from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch_geometric.data import HeteroData

from .graph_builder import HeteroGraph


def _tensor_from_df_numeric(df, columns, dtype=torch.float32):
    import numpy as np

    if not columns:
        return None
    arr = df[columns].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr).to(dtype)


def _encode_categorical(series) -> Tuple[torch.Tensor, Dict[str, int]]:
    # Build mapping and encode to integer tensor
    categories = series.fillna("<na>").astype(str).tolist()
    uniq = sorted(set(categories))
    mapping = {v: i for i, v in enumerate(uniq)}
    idx = torch.tensor([mapping[v] for v in categories], dtype=torch.long)
    return idx, mapping


def build_pyg_data(graph: HeteroGraph) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
    data = HeteroData()
    vocab: Dict[str, Dict[str, int]] = {}

    # Node features
    start_df = graph.nodes["Start_Action"]
    end_df = graph.nodes["End_Action"]
    player_df = graph.nodes["Player"]
    team_df = graph.nodes["Team"]

    # Numeric features
    data["Start_Action"].x = _tensor_from_df_numeric(
        start_df,
        ["period_id", "time_seconds", "start_x", "start_y", "is_home_team"],
    )
    data["End_Action"].x = _tensor_from_df_numeric(
        end_df,
        ["period_id", "time_seconds", "end_x", "end_y", "is_home_team"],
    )

    # Categorical indices for embeddings
    start_type_idx, start_type_map = _encode_categorical(start_df["type_name"])  # type: ignore
    start_body_idx, start_body_map = _encode_categorical(start_df["bodypart_name"])  # type: ignore
    end_type_idx, end_type_map = _encode_categorical(end_df["type_name"])  # type: ignore
    end_body_idx, end_body_map = _encode_categorical(end_df["bodypart_name"])  # type: ignore

    data["Start_Action"].type_idx = start_type_idx
    data["Start_Action"].body_idx = start_body_idx
    data["End_Action"].type_idx = end_type_idx
    data["End_Action"].body_idx = end_body_idx

    vocab["type_name"] = start_type_map  # primary mapping for labels
    vocab["bodypart_name_start"] = start_body_map
    vocab["bodypart_name_end"] = end_body_map

    # For Player and Team, use zero features; model can learn embeddings
    data["Player"].num_nodes = player_df.shape[0]
    data["Team"].num_nodes = team_df.shape[0]

    # Edges
    for (src, rel, dst), edge_index in graph.edges.items():
        data[(src, rel, dst)].edge_index = edge_index

    # Labels: for End_Action i, y = next Start_Action.type_idx if exists via followedBy
    if ("End_Action", "followedBy", "Start_Action") in data.edge_types:
        ei = data[("End_Action", "followedBy", "Start_Action")].edge_index
        num_end = end_df.shape[0]
        y = torch.full((num_end,), -1, dtype=torch.long)
        # If multiple outgoing edges exist (shouldn't), last one wins
        y[ei[0]] = data["Start_Action"].type_idx[ei[1]]  # map end node -> next start's type
        data["End_Action"].y = y

    return data, vocab


