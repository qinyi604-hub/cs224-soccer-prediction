from __future__ import annotations

from typing import Dict, Tuple, List

import torch


def _build_undirected_adj_with_edge_feats(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
) -> Tuple[List[List[int]], Dict[Tuple[int, int], torch.Tensor]]:
    """
    Build an undirected adjacency list and a map from (u, v) to edge feature vector.
    For undirected usage, we assign the same feature vector to both (u, v) and (v, u).
    """
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    feat_map: Dict[Tuple[int, int], torch.Tensor] = {}
    if edge_index.numel() == 0:
        return adj, feat_map

    i = edge_index[0].tolist()
    j = edge_index[1].tolist()
    for idx in range(len(i)):
        u = int(i[idx])
        v = int(j[idx])
        f = edge_attr[idx]
        # store for both directions; if duplicates appear, last one wins (edge_attr mirrored anyway)
        feat_map[(u, v)] = f
        feat_map[(v, u)] = f
        adj[u].append(v)
        adj[v].append(u)
    return adj, feat_map


def _bfs_shortest_path_feature_sum(
    src: int,
    max_hops: int,
    adj: List[List[int]],
    feat_map: Dict[Tuple[int, int], torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    BFS from src up to max_hops and accumulate the sum of edge features along the first discovered
    shortest path for each reachable node.
    Returns a dict: dst -> sum(feature vectors along path src->dst).
    """
    from collections import deque

    N = len(adj)
    visited = [False] * N
    parent = [-1] * N
    depth = [-1] * N
    q = deque()
    q.append(src)
    visited[src] = True
    depth[src] = 0

    while q:
        u = q.popleft()
        if depth[u] >= max_hops:
            continue
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                q.append(v)

    # For each reachable node (excluding src), reconstruct path and sum features
    sums: Dict[int, torch.Tensor] = {}
    for dst in range(N):
        if dst == src or not visited[dst]:
            continue
        # Walk back from dst to src
        cur = dst
        acc: torch.Tensor | None = None
        valid = True
        while parent[cur] != -1:
            p = parent[cur]
            feat = feat_map.get((p, cur))
            if feat is None:
                valid = False
                break
            acc = feat.clone() if acc is None else (acc + feat)
            cur = p
        if valid and cur == src and acc is not None:
            sums[dst] = acc
    return sums


def build_edge_feature_sum_matrix(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    edge_feat_dim: int,
    max_hops: int = 3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Precompute a dense [N, N, E] tensor M where M[i, j] is:
      - edge_attr(i, j) if a direct edge exists (undirected view)
      - else sum of edge_attr along a shortest path from i to j within max_hops
      - else zeros
    """
    if device is None:
        device = edge_index.device
    M = torch.zeros((num_nodes, num_nodes, edge_feat_dim), dtype=torch.float32, device=device)
    if edge_index.numel() == 0 or num_nodes == 0:
        return M

    adj, feat_map = _build_undirected_adj_with_edge_feats(edge_index, edge_attr, num_nodes)

    # Fill direct edges
    for (u, v), f in feat_map.items():
        M[u, v, :] = f.to(device)

    # BFS from each node to get shortest-path sums up to max_hops
    for src in range(num_nodes):
        sums = _bfs_shortest_path_feature_sum(src, max_hops, adj, feat_map)
        for dst, acc in sums.items():
            if torch.all(M[src, dst, :] == 0):
                M[src, dst, :] = acc.to(device)
    return M


