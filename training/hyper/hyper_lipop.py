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
            "output_embedding": "graph",
            "output_mlp": {
                "use_bias": [True, True, False], "units": [64, 32, 1],
                "activation": ["relu", "relu", "linear"]
            },
            "pooling_args": {"pooling_method": "sum"},
            "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
            "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
            "edge_activation": {"activation": "relu"},
            "node_dense": {"units": 128, "use_bias": True, "activation": "relu"},
            "verbose": 1, "depth": 5,
            "dropout": {"rate": 0.1}
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
            "KFold": {"n_splits": 5, "random_state": None, "shuffle": True},
            "execute_folds": None
        },
        "data": {
            "set_edge_indices_reverse": {}
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "1.1.0"
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
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True], "units": [200, 1],
                           "activation": ["kgcnn>leaky_relu", "linear"]},
            "attention_args": {"units": 200},
            "depth": 2,
            "dropout": 0.2,
            "verbose": 1
        },
        "training": {
            "fit": {"batch_size": 200, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW",
                              "config": {"lr": 0.0031622776601683794, "weight_decay": 1e-05
                                         }
                              }
            },
            "KFold": {"n_splits": 5, "random_state": None, "shuffle": True},
            "execute_folds": None
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "1.1.0"
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
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
            "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
            "pooling_args": {"pooling_method": "sum"}, "conv_args": {"units": 128, "cutoff": None},
            "update_args": {"units": 128}, "depth": 3, "verbose": 1
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
            "KFold": {"n_splits": 5, "random_state": None, "shuffle": True},
            "execute_folds": None
        },
        "data": {
            "set_range": {"max_distance": 3, "max_neighbours": 10000}
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "1.1.0"
        }
    },
    "GIN": {
        "model": {
            "name": "GIN",
            "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                       {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
            "input_embedding": {"node": {"input_dim": 96, "output_dim": 64}},
            "output_embedding": "graph",
            "output_mlp": {"use_bias": True, "units": [64, 32, 1], "activation": ["relu", "relu", "linear"]},
            "output_activation": "linear",
            "depth": 5,
            "dropout": 0.05,
            "gin_args": {"units": [64, 64], "use_bias": True, "activation": ["relu", "relu"]}
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
                                   "decay_rate": 0.5, "staircase":  False}
                        }
                    }
                },
                "loss": "mean_absolute_error"
            },
            "KFold": {"n_splits": 5, "random_state": None, "shuffle": True},
            "execute_folds": None
        },
        "data": {
        },
        "info": {
            "postfix" : "",
            "kgcnn_version": "1.1.0"
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
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, False], "units": [32, 32, 1],
                           "activation": ["relu", "relu", "linear"]},
            "set2set_args": {"channels": 32, "T": 3, "pooling_method": "mean", "init_qstar": "mean"},
            "node_mlp_args": {"units": [32, 32], "use_bias": True, "activation": ["relu", "linear"]},
            "edge_mlp_args": {"units": [32, 32], "activation": ["relu", "linear"]},
            "pooling_args": {"pooling_method": "segment_sum"},
            "depth": 3, "use_set2set": False, "verbose": 1,
            "gather_args": {}
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
            "KFold": {"n_splits": 5, "random_state": None, "shuffle": True},
            "execute_folds": None
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "1.1.2"
        }
    },
}
