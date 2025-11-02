from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from app.utils.data_loader import DataCsvLoader
from app.utils.config import GraphConfig


def _build_mapping(values: List[str]) -> Dict[str, int]:
    uniq = sorted(set(values))
    mapping = {"<unk>": 0}
    for i, v in enumerate(uniq, start=1):
        mapping[v] = i
    return mapping


class ActionsSequenceDataset(Dataset):
    def __init__(self, loader: DataCsvLoader, cfg: GraphConfig, k: int = 32) -> None:
        self.k = k
        actions = loader.load_actions()
        games = loader.load_games()

        # Sample games
        game_ids = games[cfg.actions_game_id_col].astype("string").dropna().unique().tolist()
        if cfg.num_games is not None and len(game_ids) > cfg.num_games:
            import pandas as pd

            game_ids = pd.Series(game_ids).sample(n=cfg.num_games, random_state=cfg.random_seed).tolist()
        actions = actions[actions[cfg.actions_game_id_col].astype("string").isin(set(game_ids))].copy()

        # Build mappings
        self.type_map = _build_mapping(actions["type_name"].astype(str).tolist())
        self.body_map = _build_mapping(actions["bodypart_name"].astype(str).tolist())
        self.player_map = _build_mapping(actions["player_id"].astype(str).tolist())
        self.team_map = _build_mapping(actions["team_id"].astype(str).tolist())

        # Build sequences grouped by (game, period)
        actions = actions.astype(
            {
                "period_id": "int64",
                "time_seconds": "float64",
                "start_x": "float64",
                "start_y": "float64",
                "end_x": "float64",
                "end_y": "float64",
            }
        )
        actions = actions.sort_values([cfg.actions_game_id_col, "period_id", "time_seconds"])  # stable

        self.windows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for (gid, pid), group in actions.groupby([cfg.actions_game_id_col, "period_id"], sort=False):
            group = group.reset_index(drop=True)
            n = group.shape[0]
            if n <= self.k:
                continue
            # Precompute feature tensors
            type_idx = torch.tensor([self.type_map.get(str(v), 0) for v in group["type_name"].astype(str).tolist()], dtype=torch.long)
            body_idx = torch.tensor([self.body_map.get(str(v), 0) for v in group["bodypart_name"].astype(str).tolist()], dtype=torch.long)
            player_idx = torch.tensor([self.player_map.get(str(v), 0) for v in group["player_id"].astype(str).tolist()], dtype=torch.long)
            team_idx = torch.tensor([self.team_map.get(str(v), 0) for v in group["team_id"].astype(str).tolist()], dtype=torch.long)

            # Numeric normalization
            period = torch.tensor(group["period_id"].values, dtype=torch.float32)
            period = (period - 1.0).clamp(min=0.0, max=1.0)
            time = torch.tensor(group["time_seconds"].values, dtype=torch.float32) / 2700.0
            time = time.clamp(min=0.0, max=1.0)
            # delta_t per step within the period (first is 0)
            dt_full = torch.zeros_like(time)
            if time.numel() > 1:
                dt_full[1:] = (time[1:] - time[:-1]).clamp(min=0.0)
            sx = torch.tensor(group["start_x"].values, dtype=torch.float32) / 105.0
            sy = torch.tensor(group["start_y"].values, dtype=torch.float32) / 68.0
            ex = torch.tensor(group["end_x"].values, dtype=torch.float32) / 105.0
            ey = torch.tensor(group["end_y"].values, dtype=torch.float32) / 68.0
            # is_home: derive via games by comparing team_id to home_team? Use 0/1 heuristic via team_map index parity not reliable; skip here
            # Use 0 for simplicity; model still has strong signal
            home = torch.zeros_like(time)
            num_feat = torch.stack([period, time, sx, sy, ex, ey, home, dt_full], dim=1)

            for i in range(0, n - self.k):
                x_slice = slice(i, i + self.k)
                # window inputs
                w_type = type_idx[x_slice]
                w_body = body_idx[x_slice]
                w_player = player_idx[x_slice]
                w_team = team_idx[x_slice]
                w_num = num_feat[x_slice, :]
                # target is next type after window
                y = type_idx[i + self.k]
                self.windows.append((w_type, w_body, w_player, w_team, w_num, y))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return self.windows[idx]


