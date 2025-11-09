from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import torch

from app.utils.config import GraphConfig
from app.utils.data_loader import DataCsvLoader


@dataclass
class HeteroGraph:
    nodes: Dict[str, pd.DataFrame]
    node_index: Dict[str, Dict[str, int]]
    edges: Dict[Tuple[str, str, str], torch.Tensor]


class HeteroGraphBuilderSingle:
    """
    Build a heterogeneous graph with a single `Action` node per actions.csv row.
    The `Action` node includes both starting and ending coordinates plus metadata.
    """

    def __init__(self, loader: DataCsvLoader, config: Optional[GraphConfig] = None) -> None:
        self.loader = loader
        self.config = config or GraphConfig()

    def _compute_is_home_team(self, actions: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
        cfg = self.config
        games_small = games[[
            cfg.actions_game_id_col,
            cfg.games_home_team_id_col,
            cfg.games_away_team_id_col,
        ]].drop_duplicates(cfg.actions_game_id_col)

        merged = actions[[cfg.actions_game_id_col, cfg.actions_team_id_col]].merge(
            games_small,
            how="left",
            left_on=cfg.actions_game_id_col,
            right_on=cfg.actions_game_id_col,
            copy=False,
        )

        is_home = (merged[cfg.actions_team_id_col] == merged[cfg.games_home_team_id_col]).astype("int64")
        return is_home.fillna(0)

    def _build_action_nodes(self, actions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        is_home = self._compute_is_home_team(actions, games)
        action_df = pd.DataFrame({
            "period_id": actions["period_id"].astype("int64"),
            "time_seconds": actions["time_seconds"].astype("float64"),
            "start_x": actions["start_x"].astype("float64"),
            "start_y": actions["start_y"].astype("float64"),
            "end_x": actions["end_x"].astype("float64"),
            "end_y": actions["end_y"].astype("float64"),
            "type_name": actions["type_name"].astype("string"),
            "result_name": actions.get("result_name", pd.Series([None]*len(actions))).astype("string"),
            "bodypart_name": actions["bodypart_name"].astype("string"),
            "is_home_team": is_home.astype("int64"),
            # carry game id for train/val split by game
            cfg.actions_game_id_col: actions[cfg.actions_game_id_col].astype("string"),
        })[cfg.action_features]
        return action_df

    def _build_player_nodes(self, players: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        cfg = self.config
        df = players[cfg.player_features + [cfg.players_id_col]].copy()
        # snake_case feature columns only
        def to_snake_case(name: str) -> str:
            out = []
            for i, ch in enumerate(name):
                if ch.isupper() and i > 0 and (name[i - 1].islower() or (i + 1 < len(name) and name[i + 1].islower())):
                    out.append("_")
                out.append(ch.lower())
            return "".join(out)
        df = df.rename(columns={c: to_snake_case(c) for c in cfg.player_features})
        df[cfg.players_id_col] = df[cfg.players_id_col].astype("string")
        id_to_idx: Dict[str, int] = {pid: i for i, pid in enumerate(df[cfg.players_id_col].tolist())}
        feature_df = df.drop(columns=[cfg.players_id_col])
        return feature_df, id_to_idx

    def _build_team_nodes(self, teams: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        cfg = self.config
        df = teams[cfg.team_features + [cfg.teams_id_col]].copy()
        def to_snake_case(name: str) -> str:
            out = []
            for i, ch in enumerate(name):
                if ch.isupper() and i > 0 and (name[i - 1].islower() or (i + 1 < len(name) and name[i + 1].islower())):
                    out.append("_")
                out.append(ch.lower())
            return "".join(out)
        df = df.rename(columns={c: to_snake_case(c) for c in cfg.team_features})
        df[cfg.teams_id_col] = df[cfg.teams_id_col].astype("string")
        id_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(df[cfg.teams_id_col].tolist())}
        feature_df = df.drop(columns=[cfg.teams_id_col])
        return feature_df, id_to_idx

    def _build_player_action_edges(self, actions: pd.DataFrame, player_id_to_idx: Dict[str, int]) -> Dict[Tuple[str, str, str], torch.Tensor]:
        cfg = self.config
        player_ids = actions[cfg.actions_player_id_col].astype("string")
        src_player_idx = []
        dst_action_idx = []
        for i, pid in enumerate(player_ids):
            idx = player_id_to_idx.get(pid)
            if idx is None:
                continue
            src_player_idx.append(idx)
            dst_action_idx.append(i)
        edge_player_to_action = torch.tensor([src_player_idx, dst_action_idx], dtype=torch.long)
        return {cfg.edge_player_to_action: edge_player_to_action}

    def _build_team_player_edges(
        self,
        players: pd.DataFrame,
        team_id_to_idx: Dict[str, int],
        player_id_to_idx: Dict[str, int],
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        cfg = self.config
        src_player = []
        dst_team = []

        def normalize_team_id_series(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            return numeric.astype("Int64").astype("string")

        player_ids = players[cfg.players_id_col].astype("string")
        if "currentTeamId" in players.columns:
            cteam_series = normalize_team_id_series(players["currentTeamId"])
            for p_id, t_id in zip(player_ids, cteam_series):
                if pd.isna(t_id):
                    continue
                t_idx = team_id_to_idx.get(str(t_id))
                p_idx = player_id_to_idx.get(str(p_id))
                if t_idx is not None and p_idx is not None:
                    src_player.append(p_idx)
                    dst_team.append(t_idx)

        if "currentNationalTeamId" in players.columns:
            cnat_series = normalize_team_id_series(players["currentNationalTeamId"])
            for p_id, t_id in zip(player_ids, cnat_series):
                if pd.isna(t_id):
                    continue
                t_idx = team_id_to_idx.get(str(t_id))
                p_idx = player_id_to_idx.get(str(p_id))
                if t_idx is not None and p_idx is not None:
                    src_player.append(p_idx)
                    dst_team.append(t_idx)

        edge_player_to_team = torch.tensor([src_player, dst_team], dtype=torch.long) if src_player else torch.empty((2, 0), dtype=torch.long)
        return {self.config.edge_player_to_team: edge_player_to_team}

    def _build_followed_by_edges(self, actions: pd.DataFrame) -> Dict[Tuple[str, str, str], torch.Tensor]:
        if actions.empty:
            return {self.config.edge_followed_by_action: torch.empty((2, 0), dtype=torch.long)}

        actions_with_index = actions[[
            self.config.actions_game_id_col,
            "period_id",
            "time_seconds",
        ]].copy()
        actions_with_index["_idx"] = actions.index

        src_idx: list[int] = []
        dst_idx: list[int] = []
        grouped = actions_with_index.sort_values([self.config.actions_game_id_col, "period_id", "time_seconds", "_idx"]).groupby([
            self.config.actions_game_id_col,
            "period_id",
        ], sort=False)
        for _, g in grouped:
            order = g["_idx"].to_list()
            if len(order) <= 1:
                continue
            src_idx.extend(order[:-1])
            dst_idx.extend(order[1:])

        edge_followed_by = torch.tensor([src_idx, dst_idx], dtype=torch.long) if src_idx else torch.empty((2, 0), dtype=torch.long)
        return {self.config.edge_followed_by_action: edge_followed_by}

    def build(self) -> HeteroGraph:
        cfg = self.config

        games_all = self.loader.load_games()
        game_ids_all = games_all[cfg.actions_game_id_col].astype("string").dropna().unique()
        if cfg.num_games is not None and len(game_ids_all) > cfg.num_games:
            rng = pd.Series(game_ids_all).sample(n=cfg.num_games, random_state=cfg.random_seed)
            selected_game_ids = set(rng.tolist())
        else:
            selected_game_ids = set(game_ids_all.tolist())
        games = games_all[games_all[cfg.actions_game_id_col].astype("string").isin(selected_game_ids)].copy()

        actions_all = self.loader.load_actions()
        actions = actions_all[actions_all[cfg.actions_game_id_col].astype("string").isin(selected_game_ids)].copy()
        actions = actions.reset_index(drop=True)

        player_ids_needed = set(actions[cfg.actions_player_id_col].astype("string").dropna().unique().tolist())
        team_ids_from_actions = set(actions[cfg.actions_team_id_col].astype("string").dropna().unique().tolist())
        home_teams = set(games[cfg.games_home_team_id_col].astype("string").dropna().unique().tolist())
        away_teams = set(games[cfg.games_away_team_id_col].astype("string").dropna().unique().tolist())

        players_all = self.loader.load_players()
        players = players_all[players_all[cfg.players_id_col].astype("string").isin(player_ids_needed)].copy()

        team_ids_from_players = set()
        if "currentTeamId" in players.columns:
            team_ids_from_players |= set(players["currentTeamId"].astype("string").dropna().unique().tolist())
        if "currentNationalTeamId" in players.columns:
            team_ids_from_players |= set(players["currentNationalTeamId"].astype("string").dropna().unique().tolist())

        team_ids_needed = team_ids_from_actions | home_teams | away_teams | team_ids_from_players
        teams_all = self.loader.load_teams()
        teams = teams_all[teams_all[cfg.teams_id_col].astype("string").isin(team_ids_needed)].copy()

        action_df = self._build_action_nodes(actions, games)
        player_df, player_id_to_idx = self._build_player_nodes(players)
        if self.config.include_unknown_player_node:
            unknown_id = self.config.unknown_player_id
            action_pids = set(actions[self.config.actions_player_id_col].astype("string").dropna().unique().tolist())
            if unknown_id in action_pids and unknown_id not in player_id_to_idx:
                new_index = len(player_df)
                player_df = player_df.reindex(player_df.index.tolist() + [new_index])
                player_id_to_idx[unknown_id] = new_index
        team_df, team_id_to_idx = self._build_team_nodes(teams)

        edges = {}
        edges.update(self._build_player_action_edges(actions, player_id_to_idx))
        edges.update(self._build_team_player_edges(players, team_id_to_idx, player_id_to_idx))
        edges.update(self._build_followed_by_edges(actions))

        nodes = {
            cfg.node_type_action: action_df,
            cfg.node_type_player: player_df,
            cfg.node_type_team: team_df,
        }
        node_index = {
            cfg.node_type_player: player_id_to_idx,
            cfg.node_type_team: team_id_to_idx,
        }
        return HeteroGraph(nodes=nodes, node_index=node_index, edges=edges)


