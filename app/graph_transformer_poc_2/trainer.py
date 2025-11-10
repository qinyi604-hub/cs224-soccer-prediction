from __future__ import annotations

from typing import Tuple, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .model import ActionGraphTransformer
from .positional import compute_action_laplacian_pos_enc
from .edge_bias import build_edge_feature_sum_matrix


def train_action_graph_transformer(
    data,
    vocab: Dict[str, Dict[str, int]],
    embed_dim: int = 128,
    heads: int = 4,
    num_layers: int = 4,
    pos_enc_dim: int = 8,
    max_path_hops: int = 3,
    k_hops: int = 3,
    seed_batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
) -> Tuple[ActionGraphTransformer, float]:
    """
    Train Action-only Graph Transformer with Laplacian positional encodings and path-based edge bias.
    Uses seed-node mini-batching: for each batch of seed Action nodes, build a k-hop Action subgraph
    (undirected neighborhood over ('Action','followedBy','Action')) and compute loss only on seeds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # Labels
    if not hasattr(data["Action"], "y"):
        raise RuntimeError("HeteroData['Action'] is missing labels 'y'. Ensure build_pyg_data created Action.y.")
    y = data["Action"].y
    mask = y >= 0
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    if idx.numel() == 0:
        raise RuntimeError("No labeled Action nodes found (y >= 0). Check edge construction for followedBy.")

    # Model
    num_action_types = len(vocab["type_name"])
    model = ActionGraphTransformer(
        num_action_types=num_action_types,
        embed_dim=embed_dim,
        num_heads=heads,
        num_layers=num_layers,
        pos_enc_dim=pos_enc_dim,
        edge_feat_dim=5,
        dropout=0.1,
        out_dim=num_action_types,
    ).to(device)

    # Build Action-only adjacency once (undirected) for k-hop neighborhoods
    if ("Action", "followedBy", "Action") not in data.edge_types:
        raise RuntimeError("Expected ('Action','followedBy','Action') relation for Action subgraph.")
    ei_full = data[("Action", "followedBy", "Action")].edge_index.to(device)
    ea_full = data[("Action", "followedBy", "Action")].edge_attr
    if ea_full is None:
        raise RuntimeError("Missing edge_attr for ('Action','followedBy','Action'); required for edge-biased attention.")
    ea_full = ea_full.to(device)
    N_total = int(data["Action"].type_idx.size(0))

    # Build undirected adjacency lists
    adj: List[List[int]] = [[] for _ in range(N_total)]
    src_list = ei_full[0].tolist()
    dst_list = ei_full[1].tolist()
    for s, d in zip(src_list, dst_list):
        s_i = int(s)
        d_i = int(d)
        adj[s_i].append(d_i)
        adj[d_i].append(s_i)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_train = int(0.8 * idx.numel())
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(idx.numel(), generator=g)
    train_idx = idx[perm[:num_train]]
    val_idx = idx[perm[num_train:]]

    type_idx_full = data["Action"].type_idx.to(device)

    train_losses = []
    val_accs = []

    def k_hop_union(seeds: List[int], k: int) -> List[int]:
        from collections import deque
        visited = set()
        dq = deque([(s, 0) for s in seeds])
        for s in seeds:
            visited.add(int(s))
        while dq:
            u, depth = dq.popleft()
            if depth >= k:
                continue
            for v in adj[int(u)]:
                if v not in visited:
                    visited.add(v)
                    dq.append((v, depth + 1))
        return sorted(list(visited))

    # Helper to build subgraph tensors from selected node ids
    def build_subgraph_tensors(selected_nodes: List[int]):
        old_to_new = {old: new for new, old in enumerate(selected_nodes)}
        # Filter edges within selected
        keep_mask = []
        for s, d in zip(src_list, dst_list):
            if s in old_to_new and d in old_to_new:
                keep_mask.append(1)
            else:
                keep_mask.append(0)
        if keep_mask:
            keep_mask_t = torch.tensor(keep_mask, dtype=torch.bool, device=device)
            ei_sub_src = ei_full[0][keep_mask_t]
            ei_sub_dst = ei_full[1][keep_mask_t]
            ea_sub = ea_full[keep_mask_t]
        else:
            ei_sub_src = torch.empty((0,), dtype=torch.long, device=device)
            ei_sub_dst = torch.empty((0,), dtype=torch.long, device=device)
            ea_sub = torch.empty((0, ea_full.size(1)), dtype=ea_full.dtype, device=device)
        if ei_sub_src.numel() > 0:
            mapped_src = torch.tensor([old_to_new[int(v.item())] for v in ei_sub_src], dtype=torch.long, device=device)
            mapped_dst = torch.tensor([old_to_new[int(v.item())] for v in ei_sub_dst], dtype=torch.long, device=device)
            ei_sub = torch.stack([mapped_src, mapped_dst], dim=0)
        else:
            ei_sub = torch.empty((2, 0), dtype=torch.long, device=device)
        num_nodes_sub = len(selected_nodes)
        type_idx_sub = type_idx_full[selected_nodes]
        y_sub = y[selected_nodes]
        lap_pos = compute_action_laplacian_pos_enc(ei_sub, num_nodes=num_nodes_sub, k=pos_enc_dim, device=device)
        edge_sums = build_edge_feature_sum_matrix(
            edge_index=ei_sub,
            edge_attr=ea_sub,
            num_nodes=num_nodes_sub,
            edge_feat_dim=ea_sub.size(1) if ea_sub.numel() > 0 else 5,
            max_hops=max_path_hops,
            device=device,
        )
        return type_idx_sub, y_sub, lap_pos, edge_sums, old_to_new

    # Training loop with seed batches
    train_idx_list = train_idx.tolist()
    val_idx_list = val_idx.tolist()

    for epoch in range(epochs):
        model.train()
        # Shuffle training seeds
        g2 = torch.Generator().manual_seed(42 + epoch)
        perm_train = torch.randperm(len(train_idx_list), generator=g2).tolist()
        shuffled = [train_idx_list[i] for i in perm_train]
        epoch_losses = []
        for start in range(0, len(shuffled), seed_batch_size):
            seeds = shuffled[start:start + seed_batch_size]
            selected_nodes = k_hop_union(seeds, k_hops)
            type_idx_sub, y_sub, lap_pos, edge_sums, old_to_new = build_subgraph_tensors(selected_nodes)
            # Map seed positions within subgraph and filter to labeled
            seed_positions = [old_to_new[s] for s in seeds if s in old_to_new]
            if not seed_positions:
                continue
            seed_positions_t = torch.tensor(seed_positions, dtype=torch.long, device=device)
            labeled_mask = (y_sub[seed_positions_t] >= 0)
            seed_positions_t = seed_positions_t[labeled_mask]
            if seed_positions_t.numel() == 0:
                continue

            optimizer.zero_grad()
            logits = model(type_idx=type_idx_sub, lap_pos_enc=lap_pos, edge_feature_sums=edge_sums)
            loss = criterion(logits[seed_positions_t], y_sub[seed_positions_t])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        train_losses.append(mean_loss)

        # Validation using seed batches from val set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(val_idx_list), seed_batch_size):
                seeds = val_idx_list[start:start + seed_batch_size]
                selected_nodes = k_hop_union(seeds, k_hops)
                type_idx_sub, y_sub, lap_pos, edge_sums, old_to_new = build_subgraph_tensors(selected_nodes)
                seed_positions = [old_to_new[s] for s in seeds if s in old_to_new]
                if not seed_positions:
                    continue
                seed_positions_t = torch.tensor(seed_positions, dtype=torch.long, device=device)
                labeled_mask = (y_sub[seed_positions_t] >= 0)
                seed_positions_t = seed_positions_t[labeled_mask]
                if seed_positions_t.numel() == 0:
                    continue
                logits = model(type_idx=type_idx_sub, lap_pos_enc=lap_pos, edge_feature_sums=edge_sums)
                pred = logits[seed_positions_t].argmax(dim=-1)
                total += int(seed_positions_t.numel())
                correct += int((pred == y_sub[seed_positions_t]).sum().item())
        val_acc = (correct / total) if total > 0 else 0.0
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} - loss: {mean_loss:.4f} - val_acc: {val_acc:.4f}")

    final_acc = val_accs[-1] if val_accs else 0.0

    # Plot metrics
    epochs_axis = list(range(1, epochs + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs_axis, train_losses, marker='o')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.plot(epochs_axis, val_accs, marker='o')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "training_result_graph_transformer_poc2.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Saved training metrics plot to {plot_path}")

    return model, final_acc


