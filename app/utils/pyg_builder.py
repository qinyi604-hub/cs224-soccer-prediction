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

    # Two modes: dual-node (Start/End) or single-node (Action)
    is_single = "Action" in graph.nodes
    if is_single:
        action_df = graph.nodes["Action"]
        player_df = graph.nodes["Player"]
        team_df = graph.nodes["Team"]
        # Numeric features for Action
        data["Action"].x = _tensor_from_df_numeric(
            action_df,
            ["period_id", "time_seconds", "start_x", "start_y", "end_x", "end_y", "is_home_team"],
        )
        # Categorical
        act_type_idx, act_type_map = _encode_categorical(action_df["type_name"])  # type: ignore
        act_body_idx, act_body_map = _encode_categorical(action_df["bodypart_name"])  # type: ignore
        data["Action"].type_idx = act_type_idx
        data["Action"].body_idx = act_body_idx
        vocab["type_name"] = act_type_map
        vocab["bodypart_name_action"] = act_body_map
        data["Player"].num_nodes = player_df.shape[0]
        data["Team"].num_nodes = team_df.shape[0]

        # Edges passthrough
        for (src, rel, dst), edge_index in graph.edges.items():
            data[(src, rel, dst)].edge_index = edge_index

        # Add reverse Team->Player so sampler can reach Team from Player at next hop
        if ("Player", "member_of", "Team") in data.edge_types:
            ei = data[("Player", "member_of", "Team")].edge_index
            data[("Team", "has_member", "Player")].edge_index = ei.flip(0)

        # Temporal edge features and reverse
        if ("Action", "followedBy", "Action") in data.edge_types:
            import numpy as np
            ei = data[("Action", "followedBy", "Action")].edge_index
            src = ei[0].cpu().numpy()
            dst = ei[1].cpu().numpy()
            t_src = action_df["time_seconds"].to_numpy(np.float32)[src]
            t_dst = action_df["time_seconds"].to_numpy(np.float32)[dst]
            delta_t = torch.from_numpy(t_dst - t_src).unsqueeze(-1).to(torch.float32)
            ex = action_df["end_x"].to_numpy(np.float32)[src]
            ey = action_df["end_y"].to_numpy(np.float32)[src]
            sx = action_df["start_x"].to_numpy(np.float32)[dst]
            sy = action_df["start_y"].to_numpy(np.float32)[dst]
            dx = torch.from_numpy(sx - ex).unsqueeze(-1).to(torch.float32)
            dy = torch.from_numpy(sy - ey).unsqueeze(-1).to(torch.float32)
            dist = torch.sqrt(dx.pow(2) + dy.pow(2))
            angle = torch.atan2(dy.squeeze(-1), dx.squeeze(-1)).unsqueeze(-1)
            edge_attr = torch.cat([delta_t, dx, dy, dist, angle], dim=1)
            data[("Action", "followedBy", "Action")].edge_attr = edge_attr
            data[("Action", "precededBy", "Action")].edge_index = ei.flip(0)
            data[("Action", "precededBy", "Action")].edge_attr = edge_attr

        # Labels: next action type via followedBy
        if ("Action", "followedBy", "Action") in data.edge_types:
            ei = data[("Action", "followedBy", "Action")].edge_index
            num_act = action_df.shape[0]
            y = torch.full((num_act,), -1, dtype=torch.long)
            y[ei[0]] = data["Action"].type_idx[ei[1]]
            data["Action"].y = y

        return data, vocab

    # Dual-node (existing) path
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

    # Edge attributes for temporal edges and reverse relation with correct semantics
    # If End_Action A -(followedBy)-> Start_Action B, then Start_Action B -(precededBy)-> End_Action A
    if ("End_Action", "followedBy", "Start_Action") in data.edge_types:
        import numpy as np

        ei = data[("End_Action", "followedBy", "Start_Action")].edge_index
        src = ei[0].cpu().numpy()
        dst = ei[1].cpu().numpy()

        t_end = end_df["time_seconds"].to_numpy(np.float32)[src]
        t_start = start_df["time_seconds"].to_numpy(np.float32)[dst]
        delta_t = torch.from_numpy(t_start - t_end).unsqueeze(-1).to(torch.float32)

        ex = end_df["end_x"].to_numpy(np.float32)[src]
        ey = end_df["end_y"].to_numpy(np.float32)[src]
        sx = start_df["start_x"].to_numpy(np.float32)[dst]
        sy = start_df["start_y"].to_numpy(np.float32)[dst]
        dx = torch.from_numpy(sx - ex).unsqueeze(-1).to(torch.float32)
        dy = torch.from_numpy(sy - ey).unsqueeze(-1).to(torch.float32)
        dist = torch.sqrt(dx.pow(2) + dy.pow(2))
        angle = torch.atan2(dy.squeeze(-1), dx.squeeze(-1)).unsqueeze(-1)
        edge_attr = torch.cat([delta_t, dx, dy, dist, angle], dim=1)

        data[("End_Action", "followedBy", "Start_Action")].edge_attr = edge_attr
        data[("Start_Action", "precededBy", "End_Action")].edge_index = ei.flip(0)
        data[("Start_Action", "precededBy", "End_Action")].edge_attr = edge_attr

    # Add intra-action edges under a distinct relation so samplers can ignore them
    N = start_df.shape[0]
    if N == end_df.shape[0] and N > 0:
        idx = torch.arange(N, dtype=torch.long)
        data[("Start_Action", "withinAction", "End_Action")].edge_index = torch.stack([idx, idx], dim=0)
        data[("End_Action", "withinAction", "Start_Action")].edge_index = torch.stack([idx, idx], dim=0)

    # Labels: for End_Action i, y = next Start_Action.type_idx if exists via followedBy
    if ("End_Action", "followedBy", "Start_Action") in data.edge_types:
        ei = data[("End_Action", "followedBy", "Start_Action")].edge_index
        num_end = end_df.shape[0]
        y = torch.full((num_end,), -1, dtype=torch.long)
        # If multiple outgoing edges exist (shouldn't), last one wins
        y[ei[0]] = data["Start_Action"].type_idx[ei[1]]  # map end node -> next start's type
        data["End_Action"].y = y

    return data, vocab


