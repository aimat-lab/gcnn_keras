hyper = {
    "GraphSAGE": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GraphSAGE",
            "config": {
                "name": "GraphSAGE",
                "inputs": [
                    {"shape": (None,), "name": "node_attributes", "dtype": "int64"},
                    {"shape": (None,), "name": "edge_attributes", "dtype": "int64"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 16},
                "node_mlp_args": {"units": [64, 32], "use_bias": True, "activation": ["relu", "linear"]},
                "edge_mlp_args": {"units": 64, "use_bias": True, "activation": "relu"},
                "pooling_args": {"pooling_method": "scatter_mean"}, "gather_args": {},
                "concat_args": {"axis": -1},
                "use_edge_features": True,
                "pooling_nodes_args": {"pooling_method": "scatter_mean"},
                "depth": 3, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "sigmoid"]}
            }
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 500, "validation_freq": 10, "verbose": 2,
                "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                               "config": {"learning_rate_start": 0.5e-3, "learning_rate_stop": 1e-5,
                                   "epo_min": 400, "epo": 500, "verbose": 0}}]
            },
            "compile": {"optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-3}},
                        "loss": "binary_crossentropy",
                        "metrics": ["binary_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "dataset": {
            "class_name": "MutagenicityDataset",
            "module_name": "kgcnn.data.datasets.MutagenicityDataset",
            "config": {},
            "methods": [
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
    "DMPNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DMPNN",
            "config": {
                "name": "DMPNN",
                "inputs": [
                    {"shape": (None,), "name": "node_attributes", "dtype": "int64"},
                    {"shape": (None,), "name": "edge_attributes", "dtype": "int64"},
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
                    "activation": ["relu", "relu", "sigmoid"]
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
                "loss": "binary_crossentropy",
                "metrics": ["binary_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "dataset": {
            "class_name": "MutagenicityDataset",
            "module_name": "kgcnn.data.datasets.MutagenicityDataset",
            "config": {},
            "methods": [
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
    "AttentiveFP": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.AttentiveFP",
            "config": {
                "name": "AttentiveFP",
                "inputs": [{"shape": [None], "name": "node_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None], "name": "edge_attributes", "dtype": "float32", "ragged": True},
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
                "class_name": "MutagenicityDataset",
                "module_name": "kgcnn.data.datasets.MutagenicityDataset",
                "config": {},
                "methods": []
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    },
    "GIN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GIN",
            "config": {
                "name": "GIN",
                "inputs": [
                    {"shape": (None, ), "name": "node_attributes", "dtype": "int64"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
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
                "output_mlp": {"activation": "sigmoid", "units": 1},
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
                "loss": "binary_crossentropy",
                "metrics": ["binary_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "dataset": {
            "class_name": "MutagenicityDataset",
            "module_name": "kgcnn.data.datasets.MutagenicityDataset",
            "config": {},
            "methods": [
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
                    {"shape": [None, ], "name": "node_attributes", "dtype": "int64"},
                    {"shape": (None, ), "name": "edge_attributes", "dtype": "int64"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 8, "output_dim": 64},
                "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                                   "activation": "kgcnn>leaky_relu", "normalize_softmax": True,
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "scatter_sum"},
                "depth": 4, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "sigmoid"]},
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
                "loss": "binary_crossentropy",
                "metrics": ["binary_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "MutagenicityDataset",
                "module_name": "kgcnn.data.datasets.MutagenicityDataset",
                "config": {},
                "methods": [
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
                    {"shape": (None, ), "name": "node_attributes", "dtype": "int64"},
                    {"shape": (None, ), "name": "edge_attributes", "dtype": "int64"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 8, "output_dim": 64},
                "attention_args": {"units": 64, "use_bias": True, "use_edge_features": True,
                                   "activation": "kgcnn>leaky_relu", "normalize_softmax": True,
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "scatter_mean"},
                "depth": 1, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                               "activation": ["relu", "relu", "sigmoid"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 2, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 0, "epo": 200,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-03}},
                "loss": "binary_crossentropy",
                "metrics": ["binary_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}}
        },
        "dataset": {
            "class_name": "MutagenicityDataset",
            "module_name": "kgcnn.data.datasets.MutagenicityDataset",
            "config": {},
            "methods": [
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
    "GCN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GCN",
            "config": {
                "name": "GCN",
                "inputs": [
                    {"shape": (None,), "name": "node_attributes", "dtype": "int64"},
                    {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": False},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 25, "output_dim": 1},
                "gcn_args": {"units": 140, "use_bias": True, "activation": "relu"},
                "depth": 5, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [140, 70, 1],
                               "activation": ["relu", "relu", "sigmoid"]},
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
                "loss": "binary_crossentropy",
                "metrics": ["binary_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "dataset": {
            "class_name": "MutagenicityDataset",
            "module_name": "kgcnn.data.datasets.MutagenicityDataset",
            "config": {},
            "methods": [
                {"map_list": {"method": "normalize_edge_weights_sym"}},
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
}