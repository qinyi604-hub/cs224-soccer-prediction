from app.pipeline.pipeline_runner import PipelineRunner


def main() -> None:
    runner = PipelineRunner()
    graph = runner.build_graph()
    meta = runner.graph_metadata(graph)

    print("Graph metadata:")
    print("Node counts:", meta["node_counts"]) 
    print("Node feature dims:", meta["node_feature_dims"]) 
    print("Edge counts:", meta["edge_counts"]) 


if __name__ == "__main__":
    main()


