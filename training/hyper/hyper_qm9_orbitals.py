target_index = [5]  # 5, 6, 7 = Homo, Lumo, Gap or combination
target_name = "HOMO"

hyper = {
    "Schnet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.Schnet",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"}
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
                "output_scaling": None
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
                "loss": "mean_absolute_error"
            },
            "scaler": {"class_name": "QMGraphLabelScaler", "config": {
                "scaler": [{"class_name": "StandardScaler",
                            "config": {"with_std": True, "with_mean": True, "copy": True}},
                           ]
            }},
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
    "PAiNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.PAiNN",
            "config": {
                "name": "PAiNN",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
                "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                "pooling_args": {"pooling_method": "sum"}, "conv_args": {"units": 128, "cutoff": None},
                "update_args": {"units": 128}, "depth": 3, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 32, "epochs": 872, "validation_freq": 10, "verbose": 2, "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 3000.0, "decay_steps": 4000000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                },
                "loss": "mean_absolute_error"
            },
            "scaler": {"class_name": "QMGraphLabelScaler", "config": {
                "scaler": [{"class_name": "StandardScaler",
                            "config": {"with_std": True, "with_mean": True, "copy": True}},
                           ]
            }},
            "multi_target_indices": target_index
        },
        "data": {
            "dataset": {
                "class_name": "QM9Dataset",
                "module_name": "kgcnn.data.datasets.QM9Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_" + target_name,
            "kgcnn_version": "2.1.0"
        }
    },
}
