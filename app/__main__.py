from app.pipeline.data_loader import DataCsvLoader


def main() -> None:
    loader = DataCsvLoader()

    actions = loader.load_actions(nrows=3)
    players = loader.load_players(nrows=3)
    teams = loader.load_teams(nrows=3)
    games = loader.load_games(nrows=3)

    print("Actions (first 3 rows):")
    print(actions)
    print()

    print("Players (first 3 rows):")
    print(players)
    print()

    print("Teams (first 3 rows):")
    print(teams)
    print()

    print("Games (first 3 rows):")
    print(games)


if __name__ == "__main__":
    main()


