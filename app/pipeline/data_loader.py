from pathlib import Path
from typing import Optional

import pandas as pd


class DataCsvLoader:
    """
    Load SPADL-related CSV datasets from the project's data directory.

    Datasets supported:
    - actions.csv
    - players.csv
    - teams.csv
    - games.csv
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[1] / "data"
        self.data_dir: Path = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load(self, csv_filename: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Generic CSV loader. Optionally limit rows with nrows for quick previews."""
        csv_path = self.data_dir / csv_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        return pd.read_csv(csv_path, nrows=nrows, low_memory=False)

    def load_actions(self, nrows: Optional[int] = None) -> pd.DataFrame:
        # Only keep SPADL columns defined in spadl_explanation.txt and enforce types
        columns = [
            "game_id",
            "period_id",
            "time_seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "type_name",
            "result_name",
            "bodypart_name",
        ]
        dtype_map = {
            "game_id": "string",
            "period_id": "int64",
            "time_seconds": "float64",
            "team_id": "string",
            "player_id": "string",
            "start_x": "float64",
            "start_y": "float64",
            "end_x": "float64",
            "end_y": "float64",
            "type_name": "string",
            "result_name": "string",
            "bodypart_name": "string",
        }
        csv_path = self.data_dir / "actions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        return pd.read_csv(
            csv_path,
            usecols=columns,
            dtype=dtype_map,
            nrows=nrows,
            low_memory=False,
        )

    def load_players(self, nrows: Optional[int] = None) -> pd.DataFrame:
        columns = [
            "passportArea",
            "weight",
            "firstName",
            "lastName",
            "currentTeamId",
            "birthDate",
            "height",
            "role",
            "wyId",
            "foot",
            "currentNationalTeamId",
        ]
        dtype_map = {
            "passportArea": "string",
            "weight": "int64",
            "firstName": "string",
            "lastName": "string",
            "currentTeamId": "string",
            "birthDate": "string",
            "height": "int64",
            "role": "string",
            "wyId": "string",
            "foot": "string",
            "currentNationalTeamId": "string",
        }
        csv_path = self.data_dir / "players.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        return pd.read_csv(
            csv_path,
            usecols=columns,
            dtype=dtype_map,
            nrows=nrows,
            low_memory=False,
        )

    def load_teams(self, nrows: Optional[int] = None) -> pd.DataFrame:
        columns = [
            "city",
            "wyId",
            "officialName",
            "area",
            "type",
        ]
        dtype_map = {
            "city": "string",
            "wyId": "string",
            "officialName": "string",
            "area": "string",
            "type": "string",
        }
        csv_path = self.data_dir / "teams.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        return pd.read_csv(
            csv_path,
            usecols=columns,
            dtype=dtype_map,
            nrows=nrows,
            low_memory=False,
        )

    def load_games(self, nrows: Optional[int] = None) -> pd.DataFrame:
        columns = [
            "game_id",
            "competition_id",
            "season_id",
            "game_date",
            "game_day",
            "home_team_id",
            "away_team_id",
        ]
        dtype_map = {
            "game_id": "string",
            "competition_id": "string",
            "season_id": "string",
            "game_date": "string",
            "game_day": "int64",
            "home_team_id": "string",
            "away_team_id": "string",
        }
        csv_path = self.data_dir / "games.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        return pd.read_csv(
            csv_path,
            usecols=columns,
            dtype=dtype_map,
            nrows=nrows,
            low_memory=False,
        )


