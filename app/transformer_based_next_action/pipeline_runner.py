from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
import matplotlib.pyplot as plt
from pathlib import Path

from app.utils.data_loader import DataCsvLoader
from app.utils.config import GraphConfig
from .dataset import ActionsSequenceDataset
from .model import GraphWindowTransformer
from app.utils.graph_builder import HeteroGraphBuilder
from app.utils.graph_builder_single_action import HeteroGraphBuilderSingle
from app.utils.pyg_builder import build_pyg_data
from app.relational_nn.models import RelGraphSAGE, SingleActionRelGraph


class TransformerRunner:
    def __init__(self, loader: Optional[DataCsvLoader] = None, cfg: Optional[GraphConfig] = None) -> None:
        self.loader = loader or DataCsvLoader()
        self.cfg = cfg or GraphConfig()

    def train(self, k: int = 32, epochs: int = 5, batch_size: int = 128, lr: float = 1e-3) -> float:
        dataset = ActionsSequenceDataset(self.loader, self.cfg, k=k)
        if len(dataset) == 0:
            print("No sequences; adjust num_games or k.")
            return 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        num_types = len(dataset.type_map)
        num_bodies = len(dataset.body_map)
        num_players = len(dataset.player_map)
        num_teams = len(dataset.team_map)

        model = GraphWindowTransformer(num_types, num_bodies, num_players, num_teams)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Simple train/val split by batches
        val_every = max(1, len(loader) // 5)
        best_acc = 0.0
        epoch_losses = []
        epoch_accs = []
        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            running_loss = 0.0
            for i, batch in enumerate(loader):
                wt, wb, wp, wtm, wn, y = batch
                batch = {
                    "type": wt.to(device),
                    "body": wb.to(device),
                    "player": wp.to(device),
                    "team": wtm.to(device),
                    "num": wn.to(device),
                    "y": y.to(device),
                }
                opt.zero_grad()
                logits = model(batch)
                loss = loss_fn(logits, batch["y"])
                loss.backward()
                opt.step()
                running_loss += float(loss.item())
                total += batch["y"].numel()
                correct += (logits.argmax(dim=-1) == batch["y"]).sum().item()
                if (i + 1) % val_every == 0:
                    acc = correct / max(1, total)
                    print(f"Epoch {epoch+1}/{epochs} step {i+1}/{len(loader)} - loss {running_loss/(i+1):.4f} acc {acc:.4f}")
            epoch_loss = running_loss / max(1, len(loader))
            epoch_acc = correct / max(1, total)
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)
            best_acc = max(best_acc, epoch_acc)
        # Save training curves
        out_dir = Path(__file__).resolve().parents[1] / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        ax1.set_title('Transformer Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax2.plot(range(1, len(epoch_accs) + 1), epoch_accs, marker='o')
        ax2.set_title('Transformer Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        out_path = out_dir / "training_result_transformer.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"Saved transformer training plot to {out_path}")
        return best_acc

    def train_graph_transformer(self, epochs: int = 3, lr: float = 1e-3, num_layers: int = 5, k_hops: int = 5, batch_size: int = 1024, graph_mode: str = "dual") -> float:
        # Build hetero graph (same as relational runner) and train a graph-transformer
        if graph_mode == "single":
            builder = HeteroGraphBuilderSingle(self.loader, self.cfg)
        else:
            builder = HeteroGraphBuilder(self.loader, self.cfg)
        graph = builder.build()
        data, vocab = build_pyg_data(graph)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)

        num_player = data["Player"].num_nodes
        num_team = data["Team"].num_nodes

        is_single = "Action" in data.node_types
        if is_single:
            num_action_types = len(vocab["type_name"])
            num_action_bodies = len(vocab["bodypart_name_action"])
            model = SingleActionRelGraph(
                metadata=data.metadata(),
                hidden_dim=64,
                out_dim=num_action_types,
                num_action_types=num_action_types,
                num_action_bodies=num_action_bodies,
                num_player_nodes=num_player,
                num_team_nodes=num_team,
                num_layers=min(num_layers, 5),
                dropout=0.1,
            ).to(device)
            target_ntype = "Action"
        else:
            num_start_types = len(vocab["type_name"])
            num_start_bodies = len(vocab["bodypart_name_start"])
            num_end_types = len(vocab["type_name"])
            num_end_bodies = len(vocab["bodypart_name_end"])
            model = RelGraphSAGE(
                metadata=data.metadata(),
                hidden_dim=64,
                out_dim=num_start_types,
                num_start_types=num_start_types,
                num_start_bodies=num_start_bodies,
                num_end_types=num_end_types,
                num_end_bodies=num_end_bodies,
                num_player_nodes=num_player,
                num_team_nodes=num_team,
                num_layers=min(num_layers, 5),
                dropout=0.1,
            ).to(device)
            target_ntype = "End_Action"

        y_full = data[target_ntype].y
        mask = y_full >= 0
        seed_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if seed_idx.numel() == 0:
            print("No labeled End_Action nodes.")
            return 0.0
        num_train = int(0.8 * seed_idx.numel())
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(seed_idx.numel(), generator=g)
        train_seeds = seed_idx[perm[:num_train]]
        val_seeds = seed_idx[perm[num_train:]]

        # Neighbor sampling restricted to temporal past only (precededBy) and ignore withinAction
        num_neighbors = {}
        for et in data.edge_types:
            if et == ("Start_Action", "precededBy", "End_Action") or et == ("Action", "precededBy", "Action"):
                num_neighbors[et] = [1] * k_hops
            else:
                num_neighbors[et] = [0] * k_hops

        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=(target_ntype, train_seeds),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=(target_ntype, val_seeds),
            batch_size=batch_size,
            shuffle=False,
        )

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        epoch_losses = []
        epoch_accs = []
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0
            for batch in train_loader:
                batch = batch.to(device)
                bs = int(batch[target_ntype].batch_size)
                if bs == 0:
                    continue
                logits = model(batch)[target_ntype][ : bs]
                y = batch[target_ntype].y[ : bs]
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += float(loss.item())
                total += y.numel()
                correct += (logits.argmax(dim=-1) == y).sum().item()

            model.eval()
            with torch.no_grad():
                v_total = 0
                v_correct = 0
                for batch in val_loader:
                    batch = batch.to(device)
                    bs = int(batch[target_ntype].batch_size)
                    if bs == 0:
                        continue
                    logits = model(batch)[target_ntype][ : bs]
                    y = batch[target_ntype].y[ : bs]
                    v_total += y.numel()
                    v_correct += (logits.argmax(dim=-1) == y).sum().item()
                acc = (v_correct / max(1, v_total))
            epoch_losses.append(running_loss / max(1, len(train_loader)))
            epoch_accs.append(acc)
            print(f"Graph-Transformer Epoch {epoch+1}/{epochs} - loss: {epoch_losses[-1]:.4f} - val_acc: {acc:.4f}")

        # Save training curves
        out_dir = Path(__file__).resolve().parents[1] / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        ax1.set_title('Graph Transformer Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax2.plot(range(1, len(epoch_accs) + 1), epoch_accs, marker='o')
        ax2.set_title('Graph Transformer Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        out_path = out_dir / "training_result_graph_transformer.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"Saved graph transformer training plot to {out_path}")
        return epoch_accs[-1] if epoch_accs else 0.0


