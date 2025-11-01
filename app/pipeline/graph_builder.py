from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import torch

from .config import GraphConfig
from .data_loader import DataCsvLoader


def to_snake_case(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (name[i - 1].islower() or (i + 1 < len(name) and name[i + 1].islower())):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


@dataclass
class HeteroGraph:
    nodes: Dict[str, pd.DataFrame]
    node_index: Dict[str, Dict[str, int]]  # maps external ID (string) -> node index per type
    edges: Dict[Tuple[str, str, str], torch.Tensor]  # (src, rel, dst) -> LongTensor [2, E]


class HeteroGraphBuilder:
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
        is_home = is_home.fillna(0)
        return is_home

    def _build_action_nodes(self, actions: pd.DataFrame, games: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config

        is_home = self._compute_is_home_team(actions, games)

        start_df = pd.DataFrame({
            "period_id": actions["period_id"].astype("int64"),
            "time_seconds": actions["time_seconds"].astype("float64"),
            "start_x": actions["start_x"].astype("float64"),
            "start_y": actions["start_y"].astype("float64"),
            "type_name": actions["type_name"].astype("string"),
            "bodypart_name": actions["bodypart_name"].astype("string"),
            "is_home_team": is_home.astype("int64"),
        })[cfg.start_action_features]

        end_df = pd.DataFrame({
            "period_id": actions["period_id"].astype("int64"),
            "time_seconds": actions["time_seconds"].astype("float64"),
            "end_x": actions["end_x"].astype("float64"),
            "end_y": actions["end_y"].astype("float64"),
            "type_name": actions["type_name"].astype("string"),
            "result_name": actions["result_name"].astype("string"),
            "bodypart_name": actions["bodypart_name"].astype("string"),
            "is_home_team": is_home.astype("int64"),
        })[cfg.end_action_features]

        return start_df, end_df

    def _build_player_nodes(self, players: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        cfg = self.config
        df = players[cfg.player_features + [cfg.players_id_col]].copy()
        # Convert camelCase to snake_case for feature columns only
        df = df.rename(columns={c: to_snake_case(c) for c in cfg.player_features})
        # Build index mapping by wyId (string)
        df[cfg.players_id_col] = df[cfg.players_id_col].astype("string")
        id_to_idx: Dict[str, int] = {pid: i for i, pid in enumerate(df[cfg.players_id_col].tolist())}
        # Drop id column from features
        feature_df = df.drop(columns=[cfg.players_id_col])
        return feature_df, id_to_idx

    def _build_team_nodes(self, teams: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        cfg = self.config
        df = teams[cfg.team_features + [cfg.teams_id_col]].copy()
        df = df.rename(columns={c: to_snake_case(c) for c in cfg.team_features})
        df[cfg.teams_id_col] = df[cfg.teams_id_col].astype("string")
        id_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(df[cfg.teams_id_col].tolist())}
        feature_df = df.drop(columns=[cfg.teams_id_col])
        return feature_df, id_to_idx

    def _build_edges(
        self,
        actions: pd.DataFrame,
        player_id_to_idx: Dict[str, int],
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        cfg = self.config

        # Player -> Start_Action and Player -> End_Action
        player_ids = actions[cfg.actions_player_id_col].astype("string")

        src_player_idx = []
        dst_start_idx = []
        dst_end_idx = []
        for i, pid in enumerate(player_ids):
            idx = player_id_to_idx.get(pid)
            if idx is None:
                continue
            src_player_idx.append(idx)
            dst_start_idx.append(i)  # start_action index aligns with action row
            dst_end_idx.append(i)    # end_action index aligns with action row

        edge_player_to_start = torch.tensor([src_player_idx, dst_start_idx], dtype=torch.long)
        edge_player_to_end = torch.tensor([src_player_idx, dst_end_idx], dtype=torch.long)

        return {
            cfg.edge_player_to_start: edge_player_to_start,
            cfg.edge_player_to_end: edge_player_to_end,
        }

    def _build_team_player_edges(
        self,
        players: pd.DataFrame,
        team_id_to_idx: Dict[str, int],
        player_id_to_idx: Dict[str, int],
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        cfg = self.config
        src_player = []
        dst_team = []

        # Normalize team IDs that may be float-like (e.g., '4502.0') to integer strings ('4502')
        def normalize_team_id_series(series: pd.Series) -> pd.Series:
            # Convert to numeric, drop NaNs, then to pandas nullable Int64, then to string
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

    def build(self) -> HeteroGraph:
        cfg = self.config

        # Load full games (small) and sample game_ids if requested
        games_all = self.loader.load_games()
        game_ids_all = games_all[cfg.actions_game_id_col].astype("string").dropna().unique()
        if cfg.num_games is not None and len(game_ids_all) > cfg.num_games:
            rng = pd.Series(game_ids_all).sample(n=cfg.num_games, random_state=cfg.random_seed)
            selected_game_ids = set(rng.tolist())
        else:
            selected_game_ids = set(game_ids_all.tolist())
        games = games_all[games_all[cfg.actions_game_id_col].astype("string").isin(selected_game_ids)].copy()

        # Load actions and filter to selected games for consistency
        actions_all = self.loader.load_actions()
        actions = actions_all[actions_all[cfg.actions_game_id_col].astype("string").isin(selected_game_ids)].copy()
        # Ensure a clean, 0..N-1 index so derived Series align by index when constructing DataFrames
        actions = actions.reset_index(drop=True)

        # Identify players and teams referenced by the filtered actions/games
        player_ids_needed = set(actions[cfg.actions_player_id_col].astype("string").dropna().unique().tolist())
        team_ids_from_actions = set(actions[cfg.actions_team_id_col].astype("string").dropna().unique().tolist())
        home_teams = set(games[cfg.games_home_team_id_col].astype("string").dropna().unique().tolist())
        away_teams = set(games[cfg.games_away_team_id_col].astype("string").dropna().unique().tolist())

        # Load and filter players
        players_all = self.loader.load_players()
        players = players_all[players_all[cfg.players_id_col].astype("string").isin(player_ids_needed)].copy()

        # Teams referenced by players
        team_ids_from_players = set()
        if "currentTeamId" in players.columns:
            team_ids_from_players |= set(players["currentTeamId"].astype("string").dropna().unique().tolist())
        if "currentNationalTeamId" in players.columns:
            team_ids_from_players |= set(players["currentNationalTeamId"].astype("string").dropna().unique().tolist())

        team_ids_needed = team_ids_from_actions | home_teams | away_teams | team_ids_from_players

        # Load and filter teams
        teams_all = self.loader.load_teams()
        teams = teams_all[teams_all[cfg.teams_id_col].astype("string").isin(team_ids_needed)].copy()

        # Build nodes
        start_df, end_df = self._build_action_nodes(actions, games)
        player_df, player_id_to_idx = self._build_player_nodes(players)
        # Optionally inject an unknown player node for actions with placeholder player_id (e.g., "0")
        if self.config.include_unknown_player_node:
            unknown_id = self.config.unknown_player_id
            action_pids = set(actions[self.config.actions_player_id_col].astype("string").dropna().unique().tolist())
            if unknown_id in action_pids and unknown_id not in player_id_to_idx:
                # Create an empty/unknown row matching player feature columns
                empty_row = {col: pd.NA for col in player_df.columns}
                player_df = pd.concat([player_df, pd.DataFrame([empty_row])], ignore_index=True)
                player_id_to_idx[unknown_id] = len(player_df) - 1
        team_df, team_id_to_idx = self._build_team_nodes(teams)

        # Edges
        edges = self._build_edges(actions, player_id_to_idx)
        edges.update(self._build_team_player_edges(players, team_id_to_idx, player_id_to_idx))

        nodes = {
            self.config.node_type_start_action: start_df,
            self.config.node_type_end_action: end_df,
            self.config.node_type_player: player_df,
            self.config.node_type_team: team_df,
        }
        node_index = {
            self.config.node_type_player: player_id_to_idx,
            self.config.node_type_team: team_id_to_idx,
        }

        return HeteroGraph(nodes=nodes, node_index=node_index, edges=edges)


