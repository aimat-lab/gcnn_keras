hyper = {
    "GCN": {
        "model": {
            "name": "GCN",
            "inputs": [
                {"shape": [None, 8710], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                "edge": {"input_dim": 10, "output_dim": 64}},
            "gcn_args": {"units": 140, "use_bias": True, "activation": "relu"},
            "depth": 3, "verbose": 1,
            "output_embedding": "node",
            "output_mlp": {"use_bias": [True, True, False], "units": [140, 70, 70],
                           "activation": ["relu", "relu", "softmax"]},
        },
        "training": {
            "fit": {
                "batch_size": 1,
                "epochs": 300,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 1e-03, "learning_rate_stop": 1e-04, "epo_min": 260, "epo": 300,
                            "verbose": 0
                        }
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
            "dataset": {"class_name": "CoraDataset", "config": {}},
            "methods": {
                "make_undirected_edges": {},
                "add_edge_self_loops": {},
                "normalize_edge_weights_sym": {}
            }
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    }
}
