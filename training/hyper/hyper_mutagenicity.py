hyper = {
    "GraphSAGE": {
        "model": {
            "name": "GraphSAGE",
            "inputs": [
                {"shape": [None], "name": "node_attributes", "dtype": "float32","ragged": True},
                {"shape": [None], "name": "edge_attributes", "dtype": "float32","ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64","ragged": True}],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64},
                "edge": {"input_dim": 5, "output_dim": 16}},
            "node_mlp_args": {"units": [64, 32], "use_bias": True, "activation": ["relu", "linear"]},
            "edge_mlp_args": {"units": 64, "use_bias": True, "activation": "relu"},
            "pooling_args": {"pooling_method": "segment_mean"}, "gather_args": {},
            "concat_args": {"axis": -1},
            "use_edge_features": True,
            "pooling_nodes_args": {"pooling_method": "mean"},
            "depth": 3, "verbose": 1,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                           "activation": ["relu", "relu", "sigmoid"]},
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 500, "validation_freq": 10, "verbose": 2,
                "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                               "config": {"learning_rate_start": 0.5e-3, "learning_rate_stop": 1e-5,
                                   "epo_min": 400, "epo": 500, "verbose": 0}}]
            },
            "compile": {"optimizer": {"class_name": "Adam", "config": {"lr": 5e-3}},
                        "loss": "binary_crossentropy", "metrics": ["accuracy"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    }
}