from app.relational_nn.pipeline_runner import PipelineRunner
from app.utils.data_loader import DataCsvLoader
from app.utils.graph_visualizer import GraphVisualizer
from app.transformer_based_next_action.pipeline_runner import TransformerRunner
from app.graph_transformer_poc_2.pipeline_runner import GraphTransformerPoC2Runner
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model pipelines (relational_nn | transformer_based_next_action | graph_transformer_poc_2)")
    parser.add_argument(
        "--model",
        type=str,
        default="relational_nn",
        help="Model to run: relational_nn | transformer_based_next_action | graph_transformer_poc_2",
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
    elif args.model == "transformer_based_next_action":
        acc = TransformerRunner().train_graph_transformer(
            epochs=50,
            lr=1e-3,
            num_layers=5,
            k_hops=5,
            batch_size=1024,
            pretrain_epochs=20,
            k_pretrain=5,
        )
        print(f"Graph Transformer validation accuracy: {acc:.3f}")
    elif args.model == "graph_transformer_poc_2":
        acc = GraphTransformerPoC2Runner().train(
            epochs=20,
            lr=1e-3,
            embed_dim=128,
            heads=4,
            num_layers=4,
            pos_enc_dim=8,
            max_path_hops=3,
        )
        print(f"Graph Transformer PoC2 validation accuracy: {acc:.3f}")
    else:
        print(f"Unknown model: {args.model}. Supported: relational_nn, transformer_based_next_action, graph_transformer_poc_2")


if __name__ == "__main__":
    main()


