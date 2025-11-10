from __future__ import annotations

from typing import Tuple

import torch


def compute_action_laplacian_pos_enc(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 8,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute Laplacian eigenvector positional encodings for the Action subgraph.
    Uses the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2} on an undirected version of the graph.
    Returns top-k non-trivial eigenvectors (excluding the smallest constant eigenvector when possible).
    """
    if device is None:
        device = edge_index.device

    if edge_index.numel() == 0 or num_nodes == 0:
        return torch.zeros((num_nodes, k), dtype=torch.float32, device=device)

    # Make undirected adjacency
    ei = edge_index.to(device)
    i, j = ei[0], ei[1]
    idx = torch.cat([torch.stack([i, j], dim=0), torch.stack([j, i], dim=0)], dim=1)
    values = torch.ones(idx.size(1), dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(idx, values, (num_nodes, num_nodes)).coalesce()

    # Degree
    deg = torch.sparse.sum(A, dim=1).to_dense()  # [N]
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    # Build dense normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    A_dense = A.to_dense()
    L = torch.eye(num_nodes, device=device) - D_inv_sqrt @ A_dense @ D_inv_sqrt

    # Eigen decomposition
    # eigh returns eigenvalues in ascending order
    evals, evecs = torch.linalg.eigh(L)  # [N], [N, N]

    # Skip the first eigenvector (constant) if possible
    start = 1 if evecs.size(1) > 1 else 0
    end = min(start + k, evecs.size(1))
    out = evecs[:, start:end]

    # Pad if fewer than k
    if out.size(1) < k:
        pad = torch.zeros((num_nodes, k - out.size(1)), dtype=out.dtype, device=device)
        out = torch.cat([out, pad], dim=1)
    return out


