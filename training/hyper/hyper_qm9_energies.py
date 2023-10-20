target_index = [11]  # 10, 11, 12, 13 = 'U0', 'U', 'H', 'G' or combination
target_name = "U"

hyper = {
    "Schnet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.Schnet",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "int32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": False},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                             "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus",
                    "cfconv_pool": "scatter_sum"
                },
                "node_pooling_args": {"pooling_method": "scatter_sum"},
                "depth": 4,
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": None,
                "output_scaling": {"name": "ExtensiveMolecularLabelScaler"}
            }
        },
        "training": {
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
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 0.0005}},
                "loss": {"class_name": "kgcnn>MeanAbsoluteError", "config": {"dtype": "float64"}},
                "metrics": [{"class_name": "MeanAbsoluteError",
                             "config": {"dtype": "float64", "name": "scaled_mean_absolute_error"}},
                            {"class_name": "RootMeanSquaredError",
                             "config": {"dtype": "float64", "name": "scaled_root_mean_squared_error"}}]
            },
            # "scaler": {"class_name": "QMGraphLabelScaler", "config": {
            #     "scaler": [{"class_name": "ExtensiveMolecularLabelScaler", "config": {}}],
            #     "atomic_number": "node_number"
            # }},
            "scaler": {"class_name": "ExtensiveMolecularLabelScaler", "config": {"atomic_number": "node_number"}},
            "multi_target_indices": target_index
        },
        "data": {
            "dataset": {
                "class_name": "QM9Dataset",
                "module_name": "kgcnn.data.datasets.QM9Dataset",
                "config": {},
                "methods": [
                    {"set_train_test_indices_k_fold": {"n_splits": 10, "random_state": 42, "shuffle": True}},
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 30}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                  "count_edges": "range_indices", "count_nodes": "node_number",
                                  "total_nodes": "total_nodes"}},
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_" + target_name,
            "kgcnn_version": "4.0.0"
        }
    },
}