from __future__ import annotations

from typing import Any, Dict, Optional
import math

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
import matplotlib.pyplot as plt
from pathlib import Path

from app.utils.data_loader import DataCsvLoader
from app.utils.config import GraphConfig
from .dataset import ActionsSequenceDataset
from .model import SequenceTransformer
from app.utils.graph_builder import HeteroGraphBuilder
from app.utils.graph_builder_single_action import HeteroGraphBuilderSingle
from app.utils.pyg_builder import build_pyg_data
from app.relational_nn.models import SingleActionRelGraph


class TransformerRunner:
    def __init__(self, loader: Optional[DataCsvLoader] = None, cfg: Optional[GraphConfig] = None) -> None:
        self.loader = loader or DataCsvLoader()
        self.cfg = cfg or GraphConfig()

    def _pretrain_sequence(self, k: int = 32, epochs: int = 20, batch_size: int = 128, lr: float = 1e-3):
        dataset = ActionsSequenceDataset(self.loader, self.cfg, k=k)
        if len(dataset) == 0:
            print("No sequences; adjust num_games or k.")
            return None, None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        num_types = len(dataset.type_map)
        num_bodies = len(dataset.body_map)
        num_players = len(dataset.player_map)
        num_teams = len(dataset.team_map)

        model = SequenceTransformer(num_types, num_bodies, num_players, num_teams, d_model=128, nhead=8, num_layers=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # Pretrain with plain CE, Adam, no weight decay, stable LR
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            running = 0.0
            total = 0
            correct = 0
            for wt, wb, wp, wtm, wn, y in loader:
                batch = {
                    "type": wt.to(device),
                    "body": wb.to(device),
                    "player": wp.to(device),
                    "team": wtm.to(device),
                    "num": wn.to(device),
                }
                y = y.to(device)
                opt.zero_grad()
                logits = model(batch)
                loss = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running += float(loss.item())
                total += y.numel()
                correct += (logits.argmax(dim=-1) == y).sum().item()
            print(f"Pretrain Seq Epoch {epoch+1}/{epochs} - loss {running/max(1,len(loader)):.4f} acc {correct/max(1,total):.4f}")
        return model, dataset

    def train_graph_transformer(self, epochs: int = 3, lr: float = 1e-3, num_layers: int = 5, k_hops: int = 5, batch_size: int = 1024, pretrain_epochs: int = 5, k_pretrain: int = 32) -> float:
        # Pretrain sequence model and transfer embeddings
        seq_model, seq_dataset = self._pretrain_sequence(k=k_pretrain, epochs=pretrain_epochs, batch_size=256, lr=lr)

        # Build single-node hetero graph and train
        builder = HeteroGraphBuilderSingle(self.loader, self.cfg)
        graph = builder.build()
        data, vocab = build_pyg_data(graph)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)

        num_player = data["Player"].num_nodes
        num_team = data["Team"].num_nodes
        num_action_types = len(vocab["type_name"])
        num_action_bodies = len(vocab["bodypart_name_action"])
        model = SingleActionRelGraph(
            metadata=data.metadata(),
            hidden_dim=128,
            out_dim=num_action_types,
            num_action_types=num_action_types,
            num_action_bodies=num_action_bodies,
            num_player_nodes=num_player,
            num_team_nodes=num_team,
            num_layers=min(num_layers, 6),
            dropout=0.3,
        ).to(device)
        target_ntype = "Action"

        # Transfer pretrained embeddings for type/body
        if seq_model is not None and seq_dataset is not None:
            type_names = sorted(vocab["type_name"].keys(), key=lambda k: vocab["type_name"][k])
            import torch as _torch
            type_src_idx = _torch.tensor([seq_dataset.type_map.get(name, 0) for name in type_names], dtype=_torch.long, device=device)
            with _torch.no_grad():
                model.action_type_emb.weight.data.copy_(seq_model.type_emb.weight.data[type_src_idx])
            body_names = sorted(vocab["bodypart_name_action"].keys(), key=lambda k: vocab["bodypart_name_action"][k])
            body_src_idx = _torch.tensor([seq_dataset.body_map.get(name, 0) for name in body_names], dtype=_torch.long, device=device)
            with _torch.no_grad():
                model.action_body_emb.weight.data.copy_(seq_model.body_emb.weight.data[body_src_idx])

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

        # Neighbor sampling: temporal past along precededBy (K hops), plus side hops for Player and Team
        num_neighbors = {}
        for et in data.edge_types:
            if et == ("Action", "precededBy", "Action"):
                num_neighbors[et] = [1] * k_hops
            elif et == ("Player", "performed", "Action"):
                # 1-hop from Action to its Player via incoming Player->Action edge
                hops = [1] + [0] * max(0, k_hops - 1)
                num_neighbors[et] = hops
            elif et == ("Team", "has_member", "Player"):
                # 2-hop: from Player frontier to Team via Team->Player reverse edge
                if k_hops >= 2:
                    hops = [0, 1] + [0] * max(0, k_hops - 2)
                else:
                    hops = [0] * k_hops
                num_neighbors[et] = hops
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

        # Graph training: revert to simple baseline (plain CE + Adam)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        num_classes_graph = len(vocab["type_name"])  # labels map to this
        loss_fn = torch.nn.CrossEntropyLoss()

        epoch_losses = []
        epoch_accs = []
        epoch_top3 = []
        epoch_macro_f1 = []
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                running_loss += float(loss.item())
                total += y.numel()
                correct += (logits.argmax(dim=-1) == y).sum().item()

            model.eval()
            with torch.no_grad():
                v_total = 0
                v_correct = 0
                v_correct_top3 = 0
                tp = torch.zeros(num_classes_graph, device=device)
                fp = torch.zeros(num_classes_graph, device=device)
                fn = torch.zeros(num_classes_graph, device=device)
                for batch in val_loader:
                    batch = batch.to(device)
                    bs = int(batch[target_ntype].batch_size)
                    if bs == 0:
                        continue
                    logits = model(batch)[target_ntype][ : bs]
                    y = batch[target_ntype].y[ : bs]
                    preds = logits.argmax(dim=-1)
                    _, topk = logits.topk(k=min(3, num_classes_graph), dim=-1)
                    v_total += y.numel()
                    v_correct += (preds == y).sum().item()
                    v_correct_top3 += (topk == y.unsqueeze(-1)).any(dim=-1).sum().item()
                    for c in range(num_classes_graph):
                        yc = (y == c)
                        pc = (preds == c)
                        tp[c] += (yc & pc).sum()
                        fp[c] += ((~yc) & pc).sum()
                        fn[c] += (yc & (~pc)).sum()
                acc = (v_correct / max(1, v_total))
                acc_top3 = (v_correct_top3 / max(1, v_total))
                f1_c = (2 * tp) / torch.clamp(2 * tp + fp + fn, min=1.0)
                macro_f1 = f1_c.mean().item()
            epoch_losses.append(running_loss / max(1, len(train_loader)))
            epoch_accs.append(acc)
            epoch_top3.append(acc_top3)
            epoch_macro_f1.append(macro_f1)
            print(
                f"Graph-Transformer Epoch {epoch+1}/{epochs} - loss: {epoch_losses[-1]:.4f} "
                f"- val_acc: {acc:.4f} - val_top3: {acc_top3:.4f} - macro_f1: {macro_f1:.4f}"
            )

            # constant LR (no scheduler)

        # Save training curves
        out_dir = Path(__file__).resolve().parents[1] / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        axs[0].set_title('Training Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].grid(True, linestyle='--', alpha=0.4)
        axs[1].plot(range(1, len(epoch_accs) + 1), epoch_accs, marker='o', label='Top-1')
        axs[1].plot(range(1, len(epoch_top3) + 1), epoch_top3, marker='s', label='Top-3')
        axs[1].set_title('Validation Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_ylim(0.0, 1.0)
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.4)
        axs[2].plot(range(1, len(epoch_macro_f1) + 1), epoch_macro_f1, marker='o', color='tab:green')
        axs[2].set_title('Validation Macro-F1')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Macro-F1')
        axs[2].set_ylim(0.0, 1.0)
        axs[2].grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        out_path = out_dir / "training_result_graph_transformer.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"Saved graph transformer training plot to {out_path}")
        return epoch_accs[-1] if epoch_accs else 0.0


