from __future__ import annotations

from typing import Optional, Tuple

import torch

from app.utils.data_loader import DataCsvLoader
from app.utils.graph_builder_single_action import HeteroGraphBuilderSingle
from app.utils.pyg_builder import build_pyg_data
from .trainer import train_action_graph_transformer


class GraphTransformerPoC2Runner:
    """
    Pipeline runner for Graph Transformer PoC v2.
    Builds a single-node Action hetero graph, converts to PyG HeteroData,
    and trains a hetero graph transformer to predict next action type.
    """

    def __init__(self, loader: Optional[DataCsvLoader] = None) -> None:
        self.loader = loader or DataCsvLoader()

    def build_graph_data(self):
        builder = HeteroGraphBuilderSingle(self.loader)
        graph = builder.build()
        data, vocab = build_pyg_data(graph)
        return data, vocab

    def train(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        embed_dim: int = 128,
        heads: int = 4,
        num_layers: int = 4,
        pos_enc_dim: int = 8,
        max_path_hops: int = 3,
        k_hops: int = 3,
        seed_batch_size: int = 64,
    ) -> float:
        data, vocab = self.build_graph_data()
        _, acc = train_action_graph_transformer(
            data=data,
            vocab=vocab,
            embed_dim=embed_dim,
            heads=heads,
            num_layers=num_layers,
            pos_enc_dim=pos_enc_dim,
            max_path_hops=max_path_hops,
            k_hops=k_hops,
            seed_batch_size=seed_batch_size,
            epochs=epochs,
            lr=lr,
        )
        return acc


