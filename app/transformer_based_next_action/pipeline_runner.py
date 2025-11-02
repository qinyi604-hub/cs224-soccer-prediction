from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from app.utils.data_loader import DataCsvLoader
from app.utils.config import GraphConfig
from .dataset import ActionsSequenceDataset
from .model import GraphWindowTransformer
from app.utils.graph_builder import HeteroGraphBuilder
from app.utils.pyg_builder import build_pyg_data
from app.relational_nn.models import RelGraphSAGE


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

    def train_graph_transformer(self, epochs: int = 3, lr: float = 1e-3) -> float:
        # Build hetero graph (same as relational runner) and train a graph-transformer
        builder = HeteroGraphBuilder(self.loader, self.cfg)
        graph = builder.build()
        data, vocab = build_pyg_data(graph)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)

        num_start_types = len(vocab["type_name"])
        num_start_bodies = len(vocab["bodypart_name_start"])
        num_end_types = len(vocab["type_name"])
        num_end_bodies = len(vocab["bodypart_name_end"])
        num_player = data["Player"].num_nodes
        num_team = data["Team"].num_nodes

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
            num_layers=2,
            dropout=0.1,
        ).to(device)

        y = data["End_Action"].y
        mask = y >= 0
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx.numel() == 0:
            print("No labeled End_Action nodes.")
            return 0.0
        num_train = int(0.8 * idx.numel())
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(idx.numel(), generator=g)
        train_idx = idx[perm[:num_train]]
        val_idx = idx[perm[num_train:]]

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        epoch_losses = []
        epoch_accs = []
        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            logits = model(data)["End_Action"]
            loss = loss_fn(logits[train_idx], y[train_idx])
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(data)["End_Action"]
                pred = logits[val_idx].argmax(dim=-1)
                acc = (pred == y[val_idx]).float().mean().item() if val_idx.numel() > 0 else 0.0
            epoch_losses.append(float(loss.item()))
            epoch_accs.append(acc)
            print(f"Graph-Transformer Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f} - val_acc: {acc:.4f}")

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


