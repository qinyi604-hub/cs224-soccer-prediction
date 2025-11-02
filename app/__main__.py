from app.pipeline.pipeline_runner import PipelineRunner
from app.pipeline.data_loader import DataCsvLoader


def main() -> None:
    # Quick verification: print one row from each loaded DataFrame
    loader = DataCsvLoader()
    print("Actions (1 row):")
    print(loader.load_actions(nrows=1))
    print()
    print("Players (1 row):")
    print(loader.load_players(nrows=1))
    print()
    print("Teams (1 row):")
    print(loader.load_teams(nrows=1))
    print()
    print("Games (1 row):")
    print(loader.load_games(nrows=1))
    print()

    runner = PipelineRunner()
    graph = runner.build_graph()
    meta = runner.graph_metadata(graph)

    print("Graph metadata:")
    print("Node counts:", meta["node_counts"]) 
    print("Node feature dims:", meta["node_feature_dims"]) 
    print("Edge counts:", meta["edge_counts"]) 


if __name__ == "__main__":
    main()


