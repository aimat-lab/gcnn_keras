hyper = {
    "GCN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GCN",
            "config": {
                "name": "GCN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 10, "output_dim": 64}},
                "gcn_args": {"units": 140, "use_bias": True, "activation": "relu"},
                "depth": 5, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [140, 70, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 800,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 5e-05, "epo_min": 250, "epo": 800,
                        "verbose": 0}
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },
}