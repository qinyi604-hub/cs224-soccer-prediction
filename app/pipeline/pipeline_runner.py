from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .data_loader import DataCsvLoader
from .graph_builder import HeteroGraph, HeteroGraphBuilder


class PipelineRunner:
    def __init__(self, loader: Optional[DataCsvLoader] = None) -> None:
        self.loader = loader or DataCsvLoader()

    def build_graph(self) -> HeteroGraph:
        builder = HeteroGraphBuilder(self.loader)
        return builder.build()

    @staticmethod
    def graph_metadata(graph: HeteroGraph) -> Dict[str, Any]:
        node_counts: Dict[str, int] = {ntype: df.shape[0] for ntype, df in graph.nodes.items()}
        node_dims: Dict[str, int] = {ntype: df.shape[1] for ntype, df in graph.nodes.items()}
        edge_counts: Dict[str, int] = {
            f"{src}__{rel}__{dst}": edge_index.shape[1] for (src, rel, dst), edge_index in graph.edges.items()
        }

        return {
            "node_counts": node_counts,
            "node_feature_dims": node_dims,
            "edge_counts": edge_counts,
            "edge_types": list(edge_counts.keys()),
        }


