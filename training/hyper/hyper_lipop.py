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
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64},
                "edge": {"input_dim": 5, "output_dim": 64}
            },
            "pooling_args": {"pooling_method": "sum"},
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
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
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
                     "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.2"
        }
    },
    "AttentiveFP": {
        "model": {
            "name": "AttentiveFP",
            "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                       {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                       {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                "edge_attributes": {"input_dim": 5, "output_dim": 64}},
            "attention_args": {"units": 200},
            "depth": 2,
            "dropout": 0.2,
            "verbose": 10,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True], "units": [200, 1],
                           "activation": ["kgcnn>leaky_relu", "linear"]},
        },
        "training": {
            "fit": {"batch_size": 200, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW",
                              "config": {"lr": 0.0031622776601683794, "weight_decay": 1e-05
                                         }
                              },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.2"
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
            "update_args": {"units": 128}, "depth": 3, "verbose": 10,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 250, "validation_freq": 10, "verbose": 2,
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
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 3, "max_neighbours": 10000}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.2"
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
            "output_mlp": {"activation": "linear", "units": 1},
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                        "class_name": "ExponentialDecay",
                        "config": {"initial_learning_rate": 0.001,
                                   "decay_steps": 5800,
                                   "decay_rate": 0.5, "staircase":  False}
                        }
                    }
                },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
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
                {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                {"shape": [], "name": "graph_size", "dtype": "float32", "ragged": False}
            ],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 32},
                                "edge": {"input_dim": 15, "output_dim": 32},
                                "graph": {"input_dim": 30, "output_dim": 32}},
            "set2set_args": {"channels": 32, "T": 3, "pooling_method": "mean", "init_qstar": "mean"},
            "node_mlp_args": {"units": [32, 32], "use_bias": True, "activation": ["relu", "linear"]},
            "edge_mlp_args": {"units": [32, 32], "activation": ["relu", "linear"]},
            "pooling_args": {"pooling_method": "segment_sum"},
            "depth": 3, "use_set2set": False, "verbose": 10,
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
            "dataset": {
                "class_name": "LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.2"
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
            "attention_heads_concat": False, "verbose": 10,
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
            "dataset": {
                "class_name": "LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.2"
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
            "attention_heads_concat": False, "verbose": 10,
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
            "dataset": {
                "class_name": "LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.2"
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
            "activation": "swish", "verbose": 10,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, False], "units": [128, 1],
                           "activation": ["swish", "linear"]},
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
            "fit": {
                "batch_size": 12, "epochs": 300, "validation_freq": 10, "verbose": 2, "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 40.0, "decay_steps": 40000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                },
                "loss": "mean_absolute_error"
            }
        },
        "data": {
            "dataset": {
                "class_name": "LipopDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 20}},
                    {"map_list": {"method": "set_angle"}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.2"
        }
    },
}
