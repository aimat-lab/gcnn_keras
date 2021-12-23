hyper = {
    "GAT": {
        "model": {
            "name": "GAT",
            "inputs": [
                    {"shape": [None, 1433], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64},
                "edge": {"input_dim": 5, "output_dim": 64}},
            "attention_args": {"units": 32, "use_bias": True, "use_edge_features": True,
                            "use_final_activation": False, "has_self_loops": True},
            "pooling_nodes_args": {"pooling_method": "mean"},
            "depth": 3, "attention_heads_num": 10,
            "attention_heads_concat": False, "verbose": 10,
            "output_embedding": "node",
            "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 7],
                           "activation": ["relu", "relu", "softmax"]},
        },
        "training": {
            "fit": {
                "batch_size": 1,
                "epochs": 250,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-04,
                        "epo_min": 200, "epo": 250, "verbose": 0}
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "categorical_crossentropy",
                "weighted_metrics": ["categorical_accuracy"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {"class_name": "CoraLuDataset", "config": {}},
            "methods": {
                "make_undirected_edges": {},
                "add_edge_self_loops": {},
                "normalize_edge_weights_sym": {}
            }
        },
        "info": {
            "postfix" : "",
            "kgcnn_version": "2.0.0"
        }
    }
}