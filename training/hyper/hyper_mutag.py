hyper = {
    "Unet": {
        "model": {
            "name": "Unet",
            "inputs": [
                {"shape": [None], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 1], "name": "edge_labels", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {
                "node": {"input_dim": 60, "output_dim": 128},
                "edge": {"input_dim": 5, "output_dim": 5}},
            "hidden_dim": {"units": 32, "use_bias": True, "activation": "linear"},
            "top_k_args": {"k": 0.3, "kernel_initializer": "ones"},
            "activation": "relu",
            "use_reconnect": True,
            "depth": 4,
            "pooling_args": {"pooling_method": "segment_mean"},
            "gather_args": {},
            "verbose": 10,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, False], "units": [25, 1], "activation": ["relu", "sigmoid"]},
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 500, "validation_freq": 2, "verbose": 2,
                    "callbacks": [
                        {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 400, "epo": 500,
                            "verbose": 0
                        }}
                    ]
                    },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-04}},
                "loss": "binary_crossentropy",
                "metrics": ["accuracy"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}}
        },
        "data": {
            "dataset": {"class_name": "MUTAGDataset", "config": {}, "methods": []},
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.2"
        }
    },
    "INorp": {
        "model": {
            "name": "INorp",
            "inputs": [
                {"shape": [None], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                {"shape": [], "name": "graph_size", "dtype": "float32", "ragged": False}],
            "input_embedding": {"node": {"input_dim": 60, "output_dim": 16},
                                "edge": {"input_dim": 4, "output_dim": 8},
                                "graph": {"input_dim": 30, "output_dim": 16}},
            "set2set_args": {"channels": 32, "T": 3, "pooling_method": "mean", "init_qstar": "mean"},
            "node_mlp_args": {"units": [16, 16], "use_bias": True, "activation": ["relu", "linear"]},
            "edge_mlp_args": {"units": [16, 16], "activation": ["relu", "linear"]},
            "pooling_args": {"pooling_method": "segment_mean"},
            "depth": 3, "use_set2set": False, "verbose": 10,
            "gather_args": {},
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, False], "units": [16, 8, 1],
                           "activation": ["relu", "relu", "sigmoid"]},
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 500, "validation_freq": 2, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 400, "epo": 500,
                        "verbose": 0
                    }
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "binary_crossentropy",
                "metrics": ["accuracy"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}}
        },
        "data": {
            "dataset": {"class_name": "MUTAGDataset", "config": {}, "methods": []},
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.2"
        }
    }
}
