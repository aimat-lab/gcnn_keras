hyper = {
    "GCN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GCN",
            "config": {
                "name": "GCN",
                "inputs": [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32"},
                    {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": True},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 25, "output_dim": 1},
                "gcn_args": {"units": 140, "use_bias": True, "activation": "relu"},
                "depth": 5, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [140, 70, 1],
                               "activation": ["relu", "relu", "linear"]},
                "output_scaling": {"name": "StandardLabelScaler"},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 800,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 5e-05, "epo_min": 250, "epo": 800,
                        "verbose": 0}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss": {"class_name": "kgcnn>MeanAbsoluteError", "config": {"dtype": "float64"}},
                "metrics": [
                    {"class_name": "MeanAbsoluteError",
                     "config": {"dtype": "float64", "name": "scaled_mean_absolute_error"}},
                    {"class_name": "RootMeanSquaredError",
                     "config": {"dtype": "float64", "name": "scaled_root_mean_squared_error"}}
                ]
            },
            "scaler": {"class_name": "StandardLabelScaler", "module_name": "kgcnn.data.transform.scaler.standard",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "dataset": {
            "class_name": "ESOLDataset",
            "module_name": "kgcnn.data.datasets.ESOLDataset",
            "config": {},
            "methods": [
                {"set_attributes": {}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "normalize_edge_weights_sym"}},
                {"map_list": {"method": "count_nodes_and_edges"}},
            ]
        },
        "data": {
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
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
                "output_embedding": "graph",
                'output_mlp': {"use_bias": [True, True], "units": [64, 1],
                               "activation": ["kgcnn>shifted_softplus", "linear"]},
                'last_mlp': {"use_bias": [True, True], "units": [128, 64],
                             "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus",
                    "cfconv_pool": "scatter_sum"
                },
                "node_pooling_args": {"pooling_method": "scatter_sum"},
                "depth": 4,
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                "output_scaling": {"name": "StandardLabelScaler"}
            }
        },
        "training": {
            "scaler": {"class_name": "StandardLabelScaler", "module_name": "kgcnn.data.transform.scaler.standard",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
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
                "metrics": [
                    {"class_name": "MeanAbsoluteError",
                     "config": {"dtype": "float64", "name": "scaled_mean_absolute_error"}},
                    {"class_name": "RootMeanSquaredError",
                     "config": {"dtype": "float64", "name": "scaled_root_mean_squared_error"}}
                ]
            }
        },
        "data": {},
        "dataset": {
            "class_name": "ESOLDataset",
            "module_name": "kgcnn.data.datasets.ESOLDataset",
            "config": {},
            "methods": [
                {"set_attributes": {}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 10000}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices"}},
            ]
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "GAT": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GAT",
            "config": {
                "name": "GAT",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 8, "output_dim": 64},
                "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                                   "activation": "kgcnn>leaky_relu",
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "scatter_sum"},
                "depth": 1, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "linear"]},
                "output_scaling": {"name": "StandardLabelScaler"},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 500, "validation_freq": 2, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 250, "epo": 500,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-03}},
                "loss": {"class_name": "kgcnn>MeanAbsoluteError", "config": {"dtype": "float64"}},
                "metrics": [
                    {"class_name": "MeanAbsoluteError",
                     "config": {"dtype": "float64", "name": "scaled_mean_absolute_error"}},
                    {"class_name": "RootMeanSquaredError",
                     "config": {"dtype": "float64", "name": "scaled_root_mean_squared_error"}}
                ]
            },
            "scaler": {"class_name": "StandardLabelScaler", "module_name": "kgcnn.data.transform.scaler.standard",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "dataset": {
            "class_name": "ESOLDataset",
            "module_name": "kgcnn.data.datasets.ESOLDataset",
            "config": {},
            "methods": [
                {"set_attributes": {}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "count_nodes_and_edges"}},
            ]
        },
        "data": {
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "GATv2": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GATv2",
            "config": {
                "name": "GATv2",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 8, "output_dim": 64},
                "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                                   "activation": "kgcnn>leaky_relu",
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "scatter_sum"},
                "depth": 4, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 500, "validation_freq": 2, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 250, "epo": 500,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-03}},
                "loss": "mean_absolute_error"
            },
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                    {"map_list": {"method": "count_nodes_and_edges"}},
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "GIN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GIN",
            "config": {
                "name": "GIN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": False},
                "input_node_embedding": {"input_dim": 96, "output_dim": 64},
                "depth": 5,
                "dropout": 0.05,
                "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                            "use_normalization": True, "normalization_technique": "graph_batch",
                            "padded_disjoint": False},
                "gin_args": {},
                "last_mlp": {"use_bias": True, "units": [64, 32, 1], "activation": ["relu", "relu", "linear"]},
                "output_embedding": "graph",
                "output_mlp": {"activation": "linear", "units": 1}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"learning_rate": {
                                  "module": "keras_core.optimizers.schedules",
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}}
                              },
                "loss": "mean_absolute_error",
            },
            "scaler": {"class_name": "StandardLabelScaler",
                       "module_name": "kgcnn.data.transform.scaler.standard",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "dataset": {
            "class_name": "ESOLDataset",
            "module_name": "kgcnn.data.datasets.ESOLDataset",
            "config": {},
            "methods": [
                {"set_attributes": {}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "count_nodes_and_edges"}},
            ]
        },
        "data": {
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "DMPNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DMPNN",
            "config": {
                "name": "DMPNN",
                "inputs": [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32"},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
                "input_graph_embedding": {"input_dim": 100, "output_dim": 64},
                "pooling_args": {"pooling_method": "scatter_sum"},
                "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
                "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
                "edge_activation": {"activation": "relu"},
                "node_dense": {"units": 128, "use_bias": True, "activation": "relu"},
                "verbose": 10, "depth": 5,
                "dropout": {"rate": 0.1},
                "output_embedding": "graph",
                "output_mlp": {
                    "use_bias": [True, True, False], "units": [64, 32, 1],
                    "activation": ["relu", "relu", "linear"]
                }
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {
                    "class_name": "Adam",
                    "config": {
                        "learning_rate":
                            {"module": "keras_core.optimizers.schedules",
                             "class_name": "ExponentialDecay",
                             "config": {"initial_learning_rate": 0.001,
                                        "decay_steps": 1600,
                                        "decay_rate": 0.5, "staircase": False}}
                    }
                },
                "loss": "mean_absolute_error",
            },
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "dataset": {
            "class_name": "ESOLDataset",
            "module_name": "kgcnn.data.datasets.ESOLDataset",
            "config": {},
            "methods": [
                {"set_attributes": {}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "set_edge_indices_reverse"}},
                {"map_list": {"method": "count_nodes_and_edges"}},
            ]
        },
        "data": {
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
}