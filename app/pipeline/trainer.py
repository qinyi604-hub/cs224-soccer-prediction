from __future__ import annotations

from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .models import RelGraphSAGE


def train_next_action_model(data, vocab, epochs: int = 100, lr: float = 1e-3) -> Tuple[RelGraphSAGE, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    num_start_types = len(vocab["type_name"])  # label space
    num_start_bodies = len(vocab["bodypart_name_start"])
    num_end_types = len(vocab["type_name"])  # reuse mapping
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Masks for End_Action nodes with valid labels
    y = data["End_Action"].y
    mask = y >= 0
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    num_train = int(0.8 * idx.numel())
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(idx.numel(), generator=g)
    train_idx = idx[perm[:num_train]]
    val_idx = idx[perm[num_train:]]

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)["End_Action"]
        loss = criterion(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            logits = model(data)["End_Action"]
            pred = logits[val_idx].argmax(dim=-1)
            acc = (pred == y[val_idx]).float().mean().item() if val_idx.numel() > 0 else 0.0
            val_accs.append(acc)
        print(f"Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f} - val_acc: {acc:.4f}")

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
    plot_path = out_dir / "training_metrics.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Saved training metrics plot to {plot_path}")

    return model, final_acc


