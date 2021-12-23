hyper = {
    "DMPNN": {
        "model": {
            "name": "DMPNN",
            "inputs": [
                {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}
            ],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                "edge": {"input_dim": 5, "output_dim": 64}},
            "pooling_args": {"pooling_method": "sum"},
            "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
            "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
            "edge_activation": {"activation": "relu"},
            "node_dense": {"units": 128, "use_bias": True, "activation": "relu"},
            "verbose": 1, "depth": 5,
            "dropout": {"rate": 0.1},
            "output_embedding": "graph",
            "output_mlp": {
                "use_bias": [True, True, False], "units": [64, 32, 1],
                "activation": ["relu", "relu", "linear"]
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}
                              }},
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {},
                "set_edge_indices_reverse": {}
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "GIN": {
        "model": {
            "name": "GIN",
            "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                       {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {"node": {"input_dim": 96, "output_dim": 64}},
            "last_mlp": {"use_bias": True, "units": [64, 32, 1], "activation": ["relu", "relu", "linear"]},
            "depth": 5,
            "dropout": 0.05,
            "gin_args": {"units": [64, 64], "use_bias": True, "activation": ["relu", "relu"],
                         "use_normalization": True, "normalization_technique": "batch"},
            "output_embedding": "graph",
            "output_mlp": {"activation": "linear", "units": 1}
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}}
                              },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {},
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "AttentiveFP": {
        "model": {
            "name": "AttentiveFP",
            "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                       {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                       {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                "edge": {"input_dim": 5, "output_dim": 64}},
            "attention_args": {"units": 200},
            "depth": 2,
            "dropout": 0.2,
            "verbose": 1,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True], "units": [200, 1],
                           "activation": ["kgcnn>leaky_relu", "linear"]}
        },
        "training": {
            "fit": {
                "batch_size": 200, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.0031622776601683794,
                                                                       "weight_decay": 1e-05}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {}
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "PAiNN": {
        "model": {
            "name": "PAiNN",
            "inputs": [
                {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
            ],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
            "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
            "pooling_args": {"pooling_method": "sum"}, "conv_args": {"units": 128, "cutoff": None},
            "update_args": {"units": 128}, "depth": 3, "verbose": 1,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 250, "validation_freq": 10, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 30.0, "decay_steps": 40000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                },
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_range": {"max_distance": 3, "max_neighbours": 10000},
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "INorp": {
        "model": {
            "name": "INorp",
            "inputs": [
                {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                {"shape": [], "name": "graph_size", "dtype": "float32", "ragged": False}
            ],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 32},
                                "edge": {"input_dim": 15, "output_dim": 32},
                                "graph": {"input_dim": 32, "output_dim": 32}},
            "set2set_args": {"channels": 32, "T": 3, "pooling_method": "mean", "init_qstar": "mean"},
            "node_mlp_args": {"units": [32, 32], "use_bias": True, "activation": ["relu", "linear"]},
            "edge_mlp_args": {"units": [32, 32], "activation": ["relu", "linear"]},
            "pooling_args": {"pooling_method": "segment_sum"},
            "depth": 3, "use_set2set": False, "verbose": 1,
            "gather_args": {},
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, False], "units": [32, 32, 1],
                           "activation": ["relu", "relu", "linear"]},
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
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {}
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "GAT": {
        "model": {
            "name": "GAT",
            "inputs": [
                {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
            ],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64},
                "edge": {"input_dim": 8, "output_dim": 64}},
            "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                               "use_final_activation": False, "has_self_loops": True},
            "pooling_nodes_args": {"pooling_method": "sum"},
            "depth": 4, "attention_heads_num": 10,
            "attention_heads_concat": False, "verbose": 1,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                           "activation": ["relu", "relu", "linear"]}
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
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {}
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "GATv2": {
        "model": {
            "name": "GATv2",
            "inputs": [
                {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
            ],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64},
                "edge": {"input_dim": 8, "output_dim": 64}},
            "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                               "use_final_activation": False, "has_self_loops": True},
            "pooling_nodes_args": {"pooling_method": "sum"},
            "depth": 4, "attention_heads_num": 10,
            "attention_heads_concat": False, "verbose": 1,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                           "activation": ["relu", "relu", "linear"]},
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
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {}
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "Schnet": {
        "model": {
            "name": "Schnet",
            "inputs": [
                {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
            ],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
            "output_embedding": "graph",
            'output_mlp': {"use_bias": [True, True], "units": [64, 1],
                           "activation": ['kgcnn>shifted_softplus', "linear"]},
            'last_mlp': {"use_bias": [True, True], "units": [128, 64],
                         "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus']},
            "interaction_args": {
                "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
            },
            "node_pooling_args": {"pooling_method": "sum"},
            "depth": 4,
            "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 1
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
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
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
                "loss": "mean_absolute_error"
            }
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_range": {"max_distance": 4, "max_neighbours": 100},
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "Megnet": {
        "model": {
            "name": "Megnet",
            "inputs": [
                {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": [2], "name": "graph_attributes", "dtype": "float32", "ragged": False}
            ],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 16},
                                "graph": {"input_dim": 100, "output_dim": 64}},
            "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
            "meg_block_args": {"node_embed": [64, 32, 32], "edge_embed": [64, 32, 32],
                               "env_embed": [64, 32, 32], "activation": "kgcnn>softplus2"},
            "set2set_args": {"channels": 16, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
            "node_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
            "edge_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
            "state_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
            "nblocks": 3, "has_ff": True, "dropout": None, "use_set2set": True,
            "make_distance": True, "expand_distance": True,
            "verbose": 1,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                           "activation": ["kgcnn>softplus2", "kgcnn>softplus2", "linear"]}
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
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
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
                "loss": "mean_absolute_error"
            }
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {},
                "set_range": {"max_distance": 4, "max_neighbours": 100},
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "GraphSAGE": {
        "model": {
            "name": "GraphSAGE",
            "inputs": [
                {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64},
                "edge": {"input_dim": 32, "output_dim": 32}},
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                           "activation": ["relu", "relu", "linear"]},
            "node_mlp_args": {"units": [64, 32], "use_bias": True, "activation": ["relu", "linear"]},
            "edge_mlp_args": {"units": 64, "use_bias": True, "activation": "relu"},
            "pooling_args": {"pooling_method": "segment_mean"}, "gather_args": {},
            "concat_args": {"axis": -1},
            "use_edge_features": True,
            "pooling_nodes_args": {"pooling_method": "sum"},
            "depth": 3, "verbose": 1
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 500, "validation_freq": 10, "verbose": 2,
                    "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                                   "config": {"learning_rate_start": 0.5e-3, "learning_rate_stop": 1e-5,
                                              "epo_min": 400, "epo": 500, "verbose": 0}}]
                    },
            "compile": {"optimizer": {"class_name": "Adam", "config": {"lr": 5e-3}},
                        "loss": "mean_absolute_error"
                        },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "set_attributes": {}
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "GCN": {
        "model": {
            "name": "GCN",
            "inputs": [
                {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                "edge": {"input_dim": 10, "output_dim": 64}},
            "gcn_args": {"units": 140, "use_bias": True, "activation": "relu"},
            "depth": 5, "verbose": 1,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, False], "units": [140, 70, 1],
                           "activation": ["relu", "relu", "linear"]},
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
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {},
                "normalize_edge_weights_sym": {}
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "DimeNetPP": {
        "model": {
            "name": "DimeNetPP",
            "inputs": [{"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                       {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                       {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                       {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                         "embeddings_initializer": {"class_name": "RandomUniform",
                                                                    "config": {"minval": -1.7320508075688772,
                                                                               "maxval": 1.7320508075688772}}}},
            "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
            "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
            "cutoff": 5.0, "envelope_exponent": 5,
            "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
            "num_targets": 128, "extensive": False, "output_init": "zeros",
            "activation": "swish", "verbose": 1,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, False], "units": [128, 1],
                           "activation": ["swish", "linear"]}
        },
        "training": {
            "fit": {
                "batch_size": 10, "epochs": 872, "validation_freq": 10, "verbose": 2, "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 30.0, "decay_steps": 40000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                },
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_range": {"max_distance": 4, "max_neighbours": 20},
                "set_angle": {},
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.0"
        }
    },
    "NMPN": {
        "model": {
            'name': "NMPN",
            'inputs': [{'shape': (None, 41), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                       {'shape': (None, 1), 'name': "edge_number", 'dtype': 'float32', 'ragged': True},
                       {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
            'input_embedding': {"node": {"input_dim": 95, "output_dim": 128},
                                "edge": {"input_dim": 5, "output_dim": 128}},
            'gauss_args': {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
            'set2set_args': {'channels': 64, 'T': 3, "pooling_method": "sum", "init_qstar": "0"},
            'pooling_args': {'pooling_method': "segment_sum"},
            'edge_mlp': {'use_bias': True, 'activation': 'swish', "units": [64, 64]},
            'use_set2set': True, 'depth': 3, 'node_dim': 128,
            "geometric_edge": False, "make_distance": False, "expand_distance": False,
            'verbose': 1,
            'output_embedding': 'graph',
            'output_mlp': {"use_bias": [True, True, False], "units": [64, 32, 1],
                           "activation": ['swish', 'swish', 'linear']},
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
                        "learning_rate_start": 1e-03, "learning_rate_stop": 5e-05, "epo_min": 100, "epo": 800,
                        "verbose": 0
                    }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {"class_name": "ESOLDataset", "config": {}},
            "methods": {
                "set_attributes": {}
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.0"
        }
    }
}
