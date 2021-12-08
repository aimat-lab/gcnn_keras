hyper = {
    "GIN": {
        "model": {
            "name": "GIN",
            "inputs": [{"shape": [None, 3], "name": "node_labels", "dtype": "float32", "ragged": True},
                      {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {"node_labels": {"input_dim": 800, "output_dim": 64}},
            "output_embedding": "graph",
            "output_mlp": {"use_bias": True, "units": 2, "activation": "softmax"},
            "last_mlp": {"use_bias": [True], "units": [2],
                         "activation": ['linear']},
            "depth": 5,
            "dropout": 0.5,
            "gin_args": {"units": [64, 64], "use_bias": True, "activation": ["relu", "relu"]}
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 150, "validation_freq": 1, "verbose": 2, "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                        "class_name": "ExponentialDecay",
                        "config": {"initial_learning_rate": 0.0005,
                                   "decay_steps": 1600,
                                   "decay_rate": 0.5, "staircase":  False}
                        }
                    }
                },
                "loss": "categorical_crossentropy"
            },
            "KFold" : {"n_splits": 5, "random_state": None, "shuffle": True},
            "execute_folds": None
        },
        "data": {
            "set_range": {"max_distance": 4, "max_neighbours": 30}
        },
        "info": {
            "postfix" : "",
            "kgcnn_version": "1.1.0"
        }
    }
}