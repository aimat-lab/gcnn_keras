hyper = {
    "Schnet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.Schnet",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64}
                },
                "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                             "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
                },
                "node_pooling_args": {"pooling_method": "sum"},
                "depth": 4,
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": None,
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 32, "epochs": 800, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}}
            },
            "scaler": {"class_name": "QMGraphLabelScaler", "config": {}},
            "multi_target_indices": 1,  # Number of states
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 30}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.0"
        }
    },
}
