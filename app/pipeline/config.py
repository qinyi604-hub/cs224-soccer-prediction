from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class GraphConfig:
    # Node type names
    node_type_start_action: str = "Start_Action"
    node_type_end_action: str = "End_Action"
    node_type_player: str = "Player"
    node_type_team: str = "Team"

    # Edge (src, relation, dst)
    edge_player_to_start: Tuple[str, str, str] = ("Player", "performed", "Start_Action")
    edge_player_to_end: Tuple[str, str, str] = ("Player", "performed", "End_Action")
    edge_player_to_team: Tuple[str, str, str] = ("Player", "member_of", "Team")

    # Feature columns to select from each table
    start_action_features: List[str] = field(
        default_factory=lambda: [
            "period_id",
            "time_seconds",
            "start_x",
            "start_y",
            "type_name",
            "bodypart_name",
            # computed feature (represents whether the action was performed by the home team)
            "is_home_team",
        ]
    )
    end_action_features: List[str] = field(
        default_factory=lambda: [
            "period_id",
            "time_seconds",
            "end_x",
            "end_y",
            "type_name",
            "result_name",
            "bodypart_name",
            # computed feature (represents whether the action was performed by the home team)
            "is_home_team",
        ]
    )
    player_features: List[str] = field(
        default_factory=lambda: [
            "passportArea",
            "weight",
            "firstName",
            "lastName",
            "currentTeamId",
            "birthDate",
            "height",
            "role",
            "foot",
        ]
    )
    team_features: List[str] = field(
        default_factory=lambda: [
            "city",
            "officialName",
            "area",
            "type",
        ]
    )

    # Identifier columns
    players_id_col: str = "wyId"
    teams_id_col: str = "wyId"
    actions_player_id_col: str = "player_id"
    actions_team_id_col: str = "team_id"
    actions_game_id_col: str = "game_id"

    # Games columns used for computing is_home_team
    games_home_team_id_col: str = "home_team_id"
    games_away_team_id_col: str = "away_team_id"

    # Sampling controls
    num_games: Optional[int] = 1
    random_seed: int = 42

    # Unknown player handling
    include_unknown_player_node: bool = True
    unknown_player_id: str = "0"

