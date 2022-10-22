hyper = {
    "GIN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GIN",
            "config": {
                "name": "GIN",
                "inputs": [{"shape": [None, 3], "name": "node_labels", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 800, "output_dim": 64}},
                "last_mlp": {"use_bias": [True], "units": [2],
                             "activation": ['linear']},
                "depth": 5,
                "dropout": 0.5,
                "gin_mlp": {"units": [64, 64], "use_bias": True, "activation": ["relu", "relu"]},
                "gin_args": {},
                "output_embedding": "graph",
                "output_mlp": {"use_bias": True, "units": 1, "activation": "sigmoid"},
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 150, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                        "class_name": "ExponentialDecay",
                        "config": {"initial_learning_rate": 0.0005,
                                   "decay_steps": 1600,
                                   "decay_rate": 0.5, "staircase":  False}
                        }
                    }
                },
                "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {"class_name": "PROTEINSDataset",
                        "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                        "config": {}, "methods": []},
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },
    "GraphSAGE": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GraphSAGE",
            "config": {
                "name": "GraphSAGE",
                "inputs": [
                    {"shape": [None, 3], "name": "node_labels", "dtype": "float32","ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32","ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64","ragged": True}],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64},
                    "edge": {"input_dim": 5, "output_dim": 16}},
                "node_mlp_args": {"units": [64, 32], "use_bias": True, "activation": ["relu", "linear"]},
                "edge_mlp_args": {"units": 64, "use_bias": True, "activation": "relu"},
                "pooling_args": {"pooling_method": "segment_mean"}, "gather_args": {},
                "concat_args": {"axis": -1},
                "use_edge_features": True,
                "pooling_nodes_args": {"pooling_method": "mean"},
                "depth": 3, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "sigmoid"]}
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 500, "validation_freq": 1, "verbose": 2,
                "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                               "config": {"learning_rate_start": 0.5e-3, "learning_rate_stop": 1e-5,
                                   "epo_min": 400, "epo": 500, "verbose": 0}}]
            },
            "compile": {"optimizer": {"class_name": "Adam", "config": {"lr": 5e-3}},
                        "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {"class_name": "PROTEINSDataset",
                        "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                        "config": {},
                        "methods": [
                            {"map_list": {"method": "normalize_edge_weights_sym"}}
                        ]},
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },
    "DMPNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DMPNN",
            "config": {
                "name": "DMPNN",
                "inputs": [
                    {"shape": [None, 3], "name": "node_labels", "dtype": "float32", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
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
                    "activation": ["relu", "relu", "sigmoid"]
                },
            }
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
                "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "PROTEINSDataset",
                "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    "CMPNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.CMPNN",
            "config": {
                "name": "CMPNN",
                "inputs": [
                    {"shape": [None, 3], "name": "node_labels", "dtype": "float32", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64}},
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
                    "activation": ["relu", "sigmoid"]
                }
            }
        },
        "training": {
            "fit": {"batch_size": 50, "epochs": 600, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}
                              }},
                "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "PROTEINSDataset",
                "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    "AttentiveFP": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.AttentiveFP",
            "config": {
                "name": "AttentiveFP",
                "inputs": [{"shape": [None, 3], "name": "node_labels", "dtype": "float32", "ragged": True},
                           {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                    "edge_attributes": {"input_dim": 5, "output_dim": 64}},
                "attention_args": {"units": 200},
                "depthato": 2, "depthmol": 3,
                "dropout": 0.2,
                "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True], "units": [200, 1],
                               "activation": ["kgcnn>leaky_relu", "sigmoid"]},
            }
        },
        "training": {
            "fit": {"batch_size": 200, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW",
                              "config": {"lr": 0.0031622776601683794, "weight_decay": 1e-05
                                         }
                              },
                "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "PROTEINSDataset",
                "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    "INorp": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.INorp",
            "config": {
                "name": "INorp",
                "inputs": [
                    {"shape": [None, 3], "name": "node_labels", "dtype": "float32", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
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
                               "activation": ["relu", "relu", "sigmoid"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 500, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 300, "epo": 500,
                        "verbose": 0}
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "PROTEINSDataset",
                "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    "GAT": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GAT",
            "config": {
                "name": "GAT",
                "inputs": [
                    {"shape": [None, 3], "name": "node_labels", "dtype": "float32", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64},
                    "edge": {"input_dim": 8, "output_dim": 64}},
                "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "sum"},
                "depth": 1, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "sigmoid"]},
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
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "PROTEINSDataset",
                "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    "GATv2": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GATv2",
            "config": {
                "name": "GATv2",
                "inputs": [
                    {"shape": [None, 3], "name": "node_labels", "dtype": "float32", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
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
                               "activation": ["relu", "relu", "sigmoid"]},
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
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "PROTEINSDataset",
                "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    "MEGAN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MEGAN",
            "config": {
                'name': "MEGAN",
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64}},
                'units': [60, 50, 40, 30],
                'importance_units': [],
                'final_units': [50, 30, 10, 1],
                "final_activation": "sigmoid",
                'dropout_rate': 0.3,
                'final_dropout_rate': 0.00,
                'importance_channels': 3,
                'return_importances': False,
                'use_edge_features': False,
                'inputs': [{'shape': (None, 3), 'name': "node_labels", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 1), 'name': "edge_weights", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
            }
        },
        "training": {
            "fit": {
                "batch_size": 64,
                "epochs": 200,
                "validation_freq": 1,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-05, "epo_min": 200, "epo": 400,
                        "verbose": 0}
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "binary_crossentropy", "metrics": ["accuracy", "AUC"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "PROTEINSDataset",
                "module_name": "kgcnn.data.datasets.PROTEINSDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.1"
        }
    },
}