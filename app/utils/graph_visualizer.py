from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from app.utils.graph_builder import HeteroGraph


class GraphVisualizer:
    def visualize_graph(self, graph: HeteroGraph, max_actions: int = 200) -> str:
        G = nx.DiGraph()

        k = min(max_actions, graph.nodes["Start_Action"].shape[0])
        action_indices = set(range(k))

        for i in range(k):
            G.add_node(("Start_Action", i), label=f"S:{i}", color="#1f77b4")
            G.add_node(("End_Action", i), label=f"E:{i}", color="#ff7f0e")

        used_players = set()
        used_teams = set()

        for (src, rel, dst), ei in graph.edges.items():
            src_idx, dst_idx = ei.tolist()
            if (src, rel, dst) == ("Player", "performed", "Start_Action"):
                for u, v in zip(src_idx, dst_idx):
                    if v in action_indices:
                        used_players.add(u)
                        G.add_node(("Player", u), label=f"P:{u}", color="#2ca02c")
                        G.add_edge(("Player", u), ("Start_Action", v), label="performed", color="#888888")
            elif (src, rel, dst) == ("Player", "performed", "End_Action"):
                for u, v in zip(src_idx, dst_idx):
                    if v in action_indices:
                        used_players.add(u)
                        G.add_node(("Player", u), label=f"P:{u}", color="#2ca02c")
                        G.add_edge(("Player", u), ("End_Action", v), label="performed", color="#888888")
            elif (src, rel, dst) == ("End_Action", "followedBy", "Start_Action"):
                for u, v in zip(src_idx, dst_idx):
                    if u in action_indices and v in action_indices:
                        G.add_edge(("End_Action", u), ("Start_Action", v), label="followedBy", color="#9467bd")
            elif (src, rel, dst) == ("Player", "member_of", "Team"):
                for u, v in zip(src_idx, dst_idx):
                    if u in used_players:
                        used_teams.add(v)
                        G.add_node(("Team", v), label=f"T:{v}", color="#d62728")
                        G.add_edge(("Player", u), ("Team", v), label="member_of", color="#17becf")

        pos = nx.spring_layout(G, seed=42, k=0.35)
        node_colors = [G.nodes[n].get("color", "#333333") for n in G.nodes]
        edge_colors = [G.edges[e].get("color", "#aaaaaa") for e in G.edges]

        plt.figure(figsize=(12, 9))
        nx.draw_networkx_nodes(G, pos, node_size=120, node_color=node_colors, linewidths=0.2, edgecolors="#222222")
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowsize=8, width=0.6, alpha=0.8)
        labels = {n: G.nodes[n]["label"] for n in G.nodes if G.nodes[n].get("label")}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

        plt.axis("off")
        out_dir = Path(__file__).resolve().parents[1] / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "graph_structure.png"
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=200)
        plt.close()
        print(f"Saved graph visualization to {out_path}")
        return str(out_path)


