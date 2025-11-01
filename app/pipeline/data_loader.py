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
        return self.load("actions.csv", nrows=nrows)

    def load_players(self, nrows: Optional[int] = None) -> pd.DataFrame:
        return self.load("players.csv", nrows=nrows)

    def load_teams(self, nrows: Optional[int] = None) -> pd.DataFrame:
        return self.load("teams.csv", nrows=nrows)

    def load_games(self, nrows: Optional[int] = None) -> pd.DataFrame:
        return self.load("games.csv", nrows=nrows)


