hyper = {
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
                "output_mlp": {"activation": "linear", "units": 1},
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"learning_rate": {
                                  "module": "keras.optimizers.schedules",
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 5800,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                              },
                "loss": "mean_absolute_error"
            },
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "dataset": {
            "class_name": "FreeSolvDataset",
            "module_name": "kgcnn.data.datasets.FreeSolvDataset",
            "config": {},
            "methods": [
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"set_attributes": {}},
                {"map_list": {"method": "count_nodes_and_edges"}},
            ]
        },
        "data": {
            "data_unit": ""
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
                                   "activation": "relu",
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "scatter_sum"},
                "depth": 1, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 500, "validation_freq": 1, "verbose": 2,
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
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "count_nodes_and_edges"}},
                ]
            },
            "data_unit": ""
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
                                   "activation": {"class_name": "function", "config": "kgcnn>leaky_relu2"},
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
                "batch_size": 32, "epochs": 500, "validation_freq": 1, "verbose": 2,
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
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
                ]
            },
            "data_unit": ""
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
                               "activation": [
                                   {"class_name": "function", "config": "kgcnn>shifted_softplus"},
                                   "linear"]
                               },
                'last_mlp': {"use_bias": [True, True], "units": [128, 64],
                             "activation": [{"class_name": "function", "config": "kgcnn>shifted_softplus"},
                                            {"class_name": "function", "config": "kgcnn>shifted_softplus"}]},
                "interaction_args": {
                    "units": 128, "use_bias": True,
                    "activation": {"class_name": "function", "config": "kgcnn>shifted_softplus"},
                    "cfconv_pool": "scatter_sum"
                },
                "node_pooling_args": {"pooling_method": "scatter_sum"},
                "depth": 4,
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
            "fit": {
                "batch_size": 32, "epochs": 800, "validation_freq": 1, "verbose": 2,
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
            }
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 10000}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                  "count_edges": "range_indices"}},
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
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
            }
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 800,
                "validation_freq": 1,
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
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
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
    "GraphSAGE": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GraphSAGE",
            "config": {
                "name": "GraphSAGE",
                "inputs": [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32"},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": False},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 32, "output_dim": 32},
                "node_mlp_args": {"units": [64, 32], "use_bias": True, "activation": ["relu", "linear"]},
                "edge_mlp_args": {"units": 64, "use_bias": True, "activation": "relu"},
                "pooling_args": {"pooling_method": "scatter_mean"}, "gather_args": {},
                "concat_args": {"axis": -1},
                "use_edge_features": True,
                "pooling_nodes_args": {"pooling_method": "scatter_sum"},
                "depth": 3, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 500, "validation_freq": 1, "verbose": 2,
                    "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                                   "config": {"learning_rate_start": 0.5e-3, "learning_rate_stop": 1e-5,
                                              "epo_min": 400, "epo": 500, "verbose": 0}}]
                    },
            "compile": {"optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-3}},
                        "loss": "mean_absolute_error"
                        },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
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
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                    {"shape": (), "name": "total_reverse", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
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
                },
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"learning_rate": {
                                  "module": "keras.optimizers.schedules",
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 5800,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                              },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "dataset": {
            "class_name": "FreeSolvDataset",
            "module_name": "kgcnn.data.datasets.FreeSolvDataset",
            "config": {},
            "methods": [
                {"set_attributes": {}},
                {"map_list": {"method": "set_edge_indices_reverse"}},
                {"map_list": {"method": "count_nodes_and_edges"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_reverse"}},
            ]
        },
        "data": {
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "CMPNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.CMPNN",
            "config": {
                "name": "CMPNN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                    {"shape": (), "name": "total_reverse", "dtype": "int64"}
                ],
                "input_tensor_type": "padded",
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
                "node_initialize": {"units": 300, "activation": "relu"},
                "edge_initialize": {"units": 300, "activation": "relu"},
                "edge_dense": {"units": 300, "activation": "linear"},
                "node_dense": {"units": 300, "activation": "linear"},
                "edge_activation": {"activation": "relu"},
                "verbose": 10,
                "depth": 5,
                "dropout": None,
                "use_final_gru": True,
                "pooling_gru": {"units": 300},
                "pooling_kwargs": {"pooling_method": "sum"},
                "output_embedding": "graph",
                "output_mlp": {
                    "use_bias": [True, False], "units": [300, 1],
                    "activation": ["relu", "linear"]
                }
            }
        },
        "training": {
            "fit": {"batch_size": 50, "epochs": 600, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"learning_rate": {
                                  "class_name": "ExponentialDecay",
                                  "module": "keras.optimizers.schedules",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}
                              }},
                "loss": "mean_squared_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}},
                    {"map_list": {"method": "count_nodes_and_edges"}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_reverse"}}
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
    "DGIN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DGIN",
            "config": {
                "name": "DGIN",
                "inputs": [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32"},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                    {"shape": (), "name": "total_reverse", "dtype": "int64"}
                ],
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
                "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                            "use_normalization": True, "normalization_technique": "graph_layer"},
                "gin_args": {},
                "pooling_args": {"pooling_method": "sum"},
                "use_graph_state": False,
                "edge_initialize": {"units": 100, "use_bias": True, "activation": "relu"},
                "edge_dense": {"units": 100, "use_bias": True, "activation": "linear"},
                "edge_activation": {"activation": "relu"},
                "node_dense": {"units": 100, "use_bias": True, "activation": "relu"},
                "verbose": 10, "depthDMPNN": 5, "depthGIN": 5,
                "dropoutDMPNN": {"rate": 0.05},
                "dropoutGIN": {"rate": 0.05},
                "output_embedding": "graph", "output_to_tensor": True,
                "last_mlp": {"use_bias": [True, True], "units": [64, 32],
                             "activation": ["relu", "relu"]},
                "output_mlp": {"use_bias": True, "units": 1,
                               "activation": "linear"}
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"learning_rate": {
                                  "module": "keras.optimizers.schedules",
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}
                              }},
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}},
                    {"map_list": {"method": "count_nodes_and_edges"}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_reverse"}},
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
    "DimeNetPP": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DimeNetPP",
            "config": {
                "name": "DimeNetPP",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "int64"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {"shape": [None, 2], "name": "angle_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"},
                    {"shape": (), "name": "total_angles", "dtype": "int64"}
                ],
                "input_tensor_type": "padded",
                "input_embedding": None,  # deprecated
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {
                    "input_dim": 95, "output_dim": 128, "embeddings_initializer": {
                        "class_name": "RandomUniform",
                        "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}
                },
                "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
                "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
                "cutoff": 5.0, "envelope_exponent": 5,
                "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
                "num_targets": 128, "extensive": False, "output_init": "zeros",
                "activation": "swish", "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, False], "units": [128, 1],
                               "activation": ["swish", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 10, "epochs": 872, "validation_freq": 10, "verbose": 2, "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Adam",
                    "config": {
                        "learning_rate": {
                            "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                "learning_rate": 0.001, "warmup_steps": 30.0, "decay_steps": 40000.0,
                                "decay_rate": 0.01
                            }
                        },
                        "use_ema": True,
                        "amsgrad": True,
                    }
                },
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "data_unit": "mol/L"
        },
        "dataset": {
            "class_name": "FreeSolvDataset",
            "module_name": "kgcnn.data.datasets.FreeSolvDataset",
            "config": {},
            "methods": [
                {"set_attributes": {}},
                {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 20}},
                {"map_list": {"method": "set_angle"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_angles",
                              "count_edges": "angle_indices"}},
            ]
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "EGNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.EGNN",
            "config": {
                "name": "EGNN",
                "inputs": [
                    {"shape": (None,), "name": "node_number", "dtype": "int64"},
                    {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
                    {"shape": (None,), "name": "edge_number", "dtype": "int64"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                ],
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 95, "output_dim": 64},
                "depth": 4,
                "node_mlp_initialize": None,
                "use_edge_attributes": True,
                "edge_mlp_kwargs": {"units": [64, 64], "activation": ["swish", "linear"]},
                "edge_attention_kwargs": None,  # {"units: 1", "activation": "sigmoid"}
                "use_normalized_difference": False,
                "expand_distance_kwargs": None,
                "coord_mlp_kwargs": {"units": [64, 1], "activation": ["swish", "linear"]},  # option: "tanh" at the end.
                "pooling_coord_kwargs": {"pooling_method": "mean"},
                "pooling_edge_kwargs": {"pooling_method": "sum"},
                "node_normalize_kwargs": None,
                "node_mlp_kwargs": {"units": [64, 64], "activation": ["swish", "linear"]},
                "use_skip": True,
                "verbose": 10,
                "node_pooling_kwargs": {"pooling_method": "sum"},
                "output_embedding": "graph",
                "output_to_tensor": None,
                "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                               "activation": ["swish", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 64, "epochs": 800, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 5e-04, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-04}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "dataset": {
            "class_name": "FreeSolvDataset",
            "module_name": "kgcnn.data.datasets.FreeSolvDataset",
            "config": {},
            "methods": [
                {"set_attributes": {}},
                {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 20}},
                {"map_list": {"method": "count_nodes_and_edges"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices"}}
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
    "GNNFilm": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GNNFilm",
            "config": {
                "name": "GNNFilm",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None], "name": "edge_number", "dtype": "int64"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                ],
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "dense_relation_kwargs": {"units": 64, "num_relations": 20},
                "dense_modulation_kwargs": {"units": 64, "num_relations": 20, "activation": "sigmoid"},
                "activation_kwargs": {"activation": "swish"},
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
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
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
    "HamNet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.HamNet",
            "config": {
                "name": "HamNet",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                ],
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
                "message_kwargs": {"units": 200,
                                   "units_edge": 200,
                                   "rate": 0.5, "use_dropout": True},
                "fingerprint_kwargs": {"units": 200,
                                       "units_attend": 200,
                                       "rate": 0.5, "use_dropout": True,
                                       "depth": 3},
                "gru_kwargs": {"units": 200},
                "verbose": 10, "depth": 3,
                "union_type_node": "gru",
                "union_type_edge": "None",
                "given_coordinates": True,
                'output_embedding': 'graph',
                'output_mlp': {"use_bias": [True, False], "units": [200, 1],
                               "activation": ['relu', 'linear'],
                               "use_dropout": [True, False],
                               "rate": [0.5, 0.0]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 40, "epochs": 400, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "AdamW",
                              "config": {"learning_rate": 0.001, "weight_decay": 1e-05}},
                "loss": "mean_squared_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
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
    "Megnet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.Megnet",
            "config": {
                "name": "Megnet",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "int64"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {"shape": [2], "name": "graph_attributes", "dtype": "float32"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"},
                ],
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 16},
                "input_graph_embedding": {"input_dim": 100, "output_dim": 64},
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
                "meg_block_args": {"node_embed": [64, 32, 32], "edge_embed": [64, 32, 32],
                                   "env_embed": [64, 32, 32],
                                   "activation": {"class_name": "function", "config": "kgcnn>softplus2"}
                                   },
                "set2set_args": {"channels": 16, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
                "node_ff_args": {"units": [64, 32],
                                 "activation": {"class_name": "function", "config": "kgcnn>softplus2"}},
                "edge_ff_args": {"units": [64, 32],
                                 "activation": {"class_name": "function", "config": "kgcnn>softplus2"}},
                "state_ff_args": {"units": [64, 32],
                                  "activation": {"class_name": "function", "config": "kgcnn>softplus2"}},
                "nblocks": 3, "has_ff": True, "dropout": None, "use_set2set": True,
                "make_distance": True, "expand_distance": True,
                "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                               "activation": [
                                   {"class_name": "function", "config": "kgcnn>softplus2"},
                                   {"class_name": "function", "config": "kgcnn>softplus2"},
                                   "linear"]}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
            "fit": {
                "batch_size": 32, "epochs": 800, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
                        "verbose": 0
                    }}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 0.0005}},
                "loss": "mean_absolute_error"
            }
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 100}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                  "count_edges": "range_indices"}}
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
    "RGCN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.RGCN",
            "config": {
                "name": "RGCN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32"},
                    {"shape": [None], "name": "edge_number", "dtype": "int64"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                ],
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "dense_relation_kwargs": {"units": 64, "num_relations": 20},
                "dense_kwargs": {"units": 64},
                "activation_kwargs": {"activation": "swish"},
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
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
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
    "HDNNP2nd": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.HDNNP2nd",
            "config": {
                "name": "HDNNP2nd",
                "inputs": [
                    {"shape": (None,), "name": "node_number", "dtype": "int64"},
                    {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
                    {"shape": (None, 2), "name": "range_indices", "dtype": "int64"},
                    {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"},
                    {"shape": (), "name": "total_angles", "dtype": "int64"}
                ],
                "input_tensor_type": "padded",
                "cast_disjoint_kwargs": {},
                "w_acsf_ang_kwargs": {},
                "w_acsf_rad_kwargs": {},
                "mlp_kwargs": {"units": [128, 128, 128, 1],
                               "num_relations": 96,
                               "activation": ["swish", "swish", "swish", "linear"]},
                "node_pooling_args": {"pooling_method": "sum"},
                "verbose": 10,
                "output_embedding": "graph", "output_to_tensor": True,
                "use_output_mlp": False,
                "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                               "activation": ["swish", "linear"]}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}},
            "fit": {
                "batch_size": 64, "epochs": 500, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 500,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 0.001}},
                "loss": "mean_absolute_error"
            }
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 8, "max_neighbours": 10000}},
                    {"map_list": {"method": "set_angle"}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                  "count_edges": "range_indices"}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_angles",
                                  "count_edges": "angle_indices"}},
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "MoGAT": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MoGAT",
            "config": {
                "name": "MoGAT",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                ],
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 64},
                "attention_args": {"units": 100},
                "depthato": 2, "depthmol": 2,
                "pooling_gat_nodes_args": {'pooling_method': 'mean'},
                "dropout": 0.2,
                "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True], "units": [1],
                               "activation": ["linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 200, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "AdamW",
                              "config": {"learning_rate": 0.001, "weight_decay": 1e-05}},
                "loss": "mean_squared_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
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
    "INorp": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.INorp",
            "config": {
                "name": "INorp",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": [], "name": "graph_size", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                ],
                'input_tensor_type': "padded",
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 32},
                "input_edge_embedding": {"input_dim": 15, "output_dim": 32},
                "input_graph_embedding": {"input_dim": 100, "output_dim": 32},
                "set2set_args": {"channels": 32, "T": 3, "pooling_method": "mean", "init_qstar": "mean"},
                "node_mlp_args": {"units": [32, 32], "use_bias": True, "activation": ["relu", "linear"]},
                "edge_mlp_args": {"units": [32, 32], "activation": ["relu", "linear"]},
                "pooling_args": {"pooling_method": "sum"},
                "depth": 3, "use_set2set": False, "verbose": 10,
                "gather_args": {},
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [32, 32, 1],
                               "activation": ["relu", "relu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 500, "validation_freq": 2, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 300, "epo": 500,
                        "verbose": 0
                    }
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
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
    "MEGAN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MEGAN",
            "config": {
                'name': "MEGAN",
                'inputs': [
                    {'shape': (None, 41), 'name': "node_attributes", 'dtype': 'float32'},
                    {'shape': (None,), 'name': "edge_number", 'dtype': 'float32'},
                    {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64'},
                    {"shape": [2], "name": "graph_attributes", "dtype": "float32"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                ],
                "input_tensor_type": "padded",
                'units': [60, 50, 40, 30],
                'importance_units': [],
                'final_units': [50, 30, 10, 1],
                'dropout_rate': 0.3,
                'final_dropout_rate': 0.00,
                'importance_channels': 3,
                'return_importances': False,
                'use_edge_features': False,
            }
        },
        "training": {
            "fit": {
                "batch_size": 64,
                "epochs": 400,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-05, "epo_min": 200, "epo": 400,
                        "verbose": 0
                    }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss": "mean_squared_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 3, "max_neighbours": 100}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
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
    "rGIN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.rGIN",
            "config": {
                "name": "rGIN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "input_tensor_type": "padded",
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 96, "output_dim": 95},
                "depth": 5,
                "dropout": 0.05,
                "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "linear"],
                            "use_normalization": True, "normalization_technique": "graph"},
                "rgin_args": {"random_range": 100},
                "last_mlp": {"use_bias": True, "units": [64, 32, 1], "activation": ["relu", "relu", "linear"]},
                "output_embedding": "graph",
                "output_mlp": {"activation": "linear", "units": 1}
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"learning_rate": {
                                  "module": "keras.optimizers.schedules",
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}}
                              },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
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
    "MXMNet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MXMNet",
            "config": {
                "name": "MXMNet",
                "inputs": [
                    {"shape": (None,), "name": "node_number", "dtype": "int64"},
                    {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
                    {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (None, 2), "name": "range_indices", "dtype": "int64"},
                    {"shape": [None, 2], "name": "angle_indices_1", "dtype": "int64"},
                    {"shape": [None, 2], "name": "angle_indices_2", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"},
                    {"shape": (), "name": "total_angles_1", "dtype": "int64"},
                    {"shape": (), "name": "total_angles_2", "dtype": "int64"}
                ],
                "input_tensor_type": "padded",
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 32},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 32},
                "bessel_basis_local": {"num_radial": 16, "cutoff": 5.0, "envelope_exponent": 5},
                "bessel_basis_global": {"num_radial": 16, "cutoff": 5.0, "envelope_exponent": 5},
                "spherical_basis_local": {"num_spherical": 7, "num_radial": 6, "cutoff": 5.0, "envelope_exponent": 5},
                "mlp_rbf_kwargs": {"units": 32, "activation": "swish"},
                "mlp_sbf_kwargs": {"units": 32, "activation": "swish"},
                "global_mp_kwargs": {"units": 32, "pooling_method": "mean"},
                "local_mp_kwargs": {"units": 32, "output_units": 1,
                                    "output_kernel_initializer": "glorot_uniform"},
                "use_edge_attributes": False,
                "depth": 4,
                "verbose": 10,
                "node_pooling_args": {"pooling_method": "sum"},
                "output_embedding": "graph", "output_to_tensor": True,
                "use_output_mlp": False,
                "output_mlp": {"use_bias": [True], "units": [1],
                               "activation": ["linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 128, "epochs": 900, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.9961697, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 45}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03, "global_clipnorm": 1000}},
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_edge_weights_uniform"}},
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 1000}},
                    {"map_list": {"method": "set_angle", "range_indices": "edge_indices", "edge_pairing": "jk",
                                  "angle_indices": "angle_indices_1",
                                  "angle_indices_nodes": "angle_indices_nodes_1",
                                  "angle_attributes": "angle_attributes_1"}},
                    {"map_list": {"method": "set_angle", "range_indices": "edge_indices", "edge_pairing": "ik",
                                  "allow_self_edges": True,
                                  "angle_indices": "angle_indices_2",
                                  "angle_indices_nodes": "angle_indices_nodes_2",
                                  "angle_attributes": "angle_attributes_2"}},
                    {"map_list": {"method": "count_nodes_and_edges"}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                  "count_edges": "range_indices"}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_angles_1",
                                  "count_edges": "angle_indices_1"}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_angles_2",
                                  "count_edges": "angle_indices_2"}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "MAT": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MAT",
            "config": {
                "name": "MAT",
                "inputs": [
                    {"shape": (None,), "name": "node_number", "dtype": "int64"},
                    {"shape": (None, 3), "name": "node_coordinates", "dtype": "float64"},
                    {"shape": (None, None, 11), "name": "adjacency_matrix", "dtype": "float64"},
                    {"shape": (None,), "name": "node_mask", "dtype": "bool"},
                    {"shape": (None, None), "name": "adjacency_mask", "dtype": "bool"},
                ],
                "input_embedding": None,
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": None,
                "distance_matrix_kwargs": {"trafo": "exp"},
                "attention_kwargs": {"units": 8, "lambda_attention": 0.3, "lambda_distance": 0.3,
                                     "lambda_adjacency": None, "add_identity": False,
                                     "dropout": 0.1},
                "feed_forward_kwargs": {"units": [32, 32, 32], "activation": ["relu", "relu", "linear"]},
                "embedding_units": 32,
                "depth": 5,
                "heads": 8,
                "merge_heads": "concat",
                "verbose": 10,
                "pooling_kwargs": {"pooling_method": "sum"},
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                               "activation": ["relu", "relu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 400,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler",
                        "config": {
                            "learning_rate_start": 5e-04, "learning_rate_stop": 1e-05, "epo_min": 0, "epo": 400,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-04}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardLabelScaler",
                       "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "FreeSolvDataset",
                "module_name": "kgcnn.data.datasets.FreeSolvDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_weights_uniform"}},
                    {"map_list": {"method": "make_dense_adjacency_matrix"}},
                    {"map_list": {"method": "make_mask", "target_property": "node_number",
                                  "mask_name": "node_mask", "rank": 1}},
                    {"map_list": {"method": "make_mask", "target_property": "adjacency_matrix",
                                  "mask_name": "adjacency_mask", "rank": 2}}
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
}
