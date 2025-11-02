from app.relational_nn.pipeline_runner import PipelineRunner
from app.utils.data_loader import DataCsvLoader
from app.utils.graph_visualizer import GraphVisualizer
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model pipelines")
    parser.add_argument(
        "--model",
        type=str,
        default="relational_nn",
        help="Model to run (e.g., relational_nn)",
    )
    args = parser.parse_args(sys.argv[1:])

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

    if args.model == "relational_nn":
        runner = PipelineRunner()
        graph = runner.build_graph()
        meta = runner.graph_metadata(graph)

        print("Graph metadata:")
        print("Node counts:", meta["node_counts"]) 
        print("Node feature dims:", meta["node_feature_dims"]) 
        print("Edge counts:", meta["edge_counts"]) 
        print()
        # Quick training demo for next action type prediction
        runner.train_demo()
        # Graph visualization (small subset)
        GraphVisualizer().visualize_graph(graph, max_actions=150)
    else:
        print(f"Unknown model: {args.model}. Supported: relational_nn")


if __name__ == "__main__":
    main()


