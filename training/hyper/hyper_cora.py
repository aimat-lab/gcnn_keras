hyper = {
    "GCN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GCN",
            "config": {
                "name": "GCN",
                "inputs": [
                    {"shape": [None, 8710], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": False,
                                         "static_batched_node_output_shape": (19793, 70)},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 25, "output_dim": 1},
                "gcn_args": {"units": 140, "use_bias": True, "activation": "relu"},
                "depth": 3, "verbose": 10,
                "output_embedding": "node",
                "output_mlp": {"use_bias": [True, True, False], "units": [140, 70, 70],
                               "activation": ["relu", "relu", "softmax"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 1,
                "epochs": 300,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 1e-03, "learning_rate_stop": 1e-04, "epo_min": 260, "epo": 300,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss": "categorical_crossentropy",
                "weighted_metrics": ["categorical_accuracy", {"class_name": "AUC", "config":{"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "CoraDataset",
                "module_name": "kgcnn.data.datasets.CoraDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "make_undirected_edges"}},
                    {"map_list": {"method": "add_edge_self_loops"}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
                    {"map_list": {"method": "count_nodes_and_edges"}},
                ]
            },
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
                    {"shape": [None, 8710], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"static_batched_node_output_shape": (19793, 70)},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 25, "output_dim": 64},
                "attention_args": {"units": 140, "use_bias": True, "use_edge_features": True,
                                    "activation": "kgcnn>leaky_relu",
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "scatter_mean"},
                "depth": 1, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "node",
                "output_mlp": {"use_bias": [True, True, False], "units": [70, 70, 70],
                               "activation": ["relu", "relu", "softmax"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 1,
                "epochs": 1000,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 5e-03, "learning_rate_stop": 1e-05,
                        "epo_min": 800, "epo": 1000, "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-03}},
                "loss": "categorical_crossentropy",
                "weighted_metrics": ["categorical_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "CoraDataset",
                "module_name": "kgcnn.data.datasets.CoraDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "make_undirected_edges"}},
                    {"map_list": {"method": "add_edge_self_loops"}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
                    {"map_list": {"method": "count_nodes_and_edges"}},
                ]
            },
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
                    {"shape": [None, 8710], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"static_batched_node_output_shape": (19793, 70)},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 8, "output_dim": 64},
                "attention_args": {"units": 70, "use_bias": True, "use_edge_features": True, "activation": "relu",
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "scatter_mean"},
                "depth": 1, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "node",
                "output_mlp": {"use_bias": [True, True, False], "units": [70, 70, 70],
                               "activation": ["relu", "relu", "softmax"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 1,
                "epochs": 250,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-05,
                        "epo_min": 200, "epo": 250, "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss": "categorical_crossentropy",
                "weighted_metrics": ["categorical_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "CoraDataset",
                "module_name": "kgcnn.data.datasets.CoraDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "make_undirected_edges"}},
                    {"map_list": {"method": "add_edge_self_loops"}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
                ]
            },
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
                    {"shape": [None, 8710], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": False, "static_batched_node_output_shape": (19793, 70)},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "input_edge_embedding": {"input_dim": 25, "output_dim": 1},
                "node_mlp_args": {"units": [70, 70], "use_bias": True, "activation": ["relu", "linear"]},
                "edge_mlp_args": {"units": 70, "use_bias": True, "activation": "relu"},
                "pooling_args": {"pooling_method": "scatter_sum"},
                "gather_args": {},
                "concat_args": {"axis": -1},
                "use_edge_features": False,
                "pooling_nodes_args": {"pooling_method": "scatter_mean"},
                "depth": 3, "verbose": 10,
                "output_embedding": "node",
                "output_mlp": {"use_bias": [True, True, False], "units": [70, 70, 70],
                               "activation": ["relu", "relu", "softmax"]}
            }
        },
        "training": {
            "fit": {"batch_size": 1, "epochs": 600, "validation_freq": 10, "verbose": 2,
                "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                               "config": {"learning_rate_start": 0.5e-3, "learning_rate_stop": 1e-5,
                                   "epo_min": 400, "epo": 600, "verbose": 0}}]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 5e-3}},
                "loss": "categorical_crossentropy",
                "weighted_metrics": ["categorical_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "CoraDataset",
                "module_name": "kgcnn.data.datasets.CoraDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "make_undirected_edges"}},
                    {"map_list": {"method": "add_edge_self_loops"}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
                ]
            },
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
                    {"shape": [None, 8710], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": False, "static_batched_node_output_shape": (19793, 70)},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "depth": 4,
                "dropout": 0.01,
                "gin_mlp": {"units": [140, 140], "use_bias": True, "activation": ["relu", "linear"],
                            "use_normalization": False, "normalization_technique": "graph_layer",
                            "padded_disjoint": False},
                "gin_args": {"trainable": True},
                "last_mlp": {"use_bias": True, "units": [140, 70, 70], "activation": ["relu", "relu", "linear"]},
                "output_embedding": "node",
                "output_mlp": {"activation": ["softmax"], "units": [70]}
            }
        },
        "training": {
            "fit": {"batch_size": 1, "epochs": 800, "validation_freq": 10, "verbose": 2,
                    "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                                   "config": {"learning_rate_start": 1e-3, "learning_rate_stop": 1e-5,
                                       "epo_min": 0, "epo": 800, "verbose": 0}}]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-3}},
                "loss": "categorical_crossentropy",
                "weighted_metrics": ["categorical_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "CoraDataset",
                "module_name": "kgcnn.data.datasets.CoraDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "make_undirected_edges"}},
                    {"map_list": {"method": "add_edge_self_loops"}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}},
                    {"map_list": {"method": "count_nodes_and_edges"}}
                ]
            },
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
                    {"shape": [None, 8710], "name": "node_attributes", "dtype": "float32"},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32"},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64"},
                    {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_edges", "dtype": "int64"},
                    {"shape": (), "name": "total_reverse", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"static_batched_node_output_shape": (19793, 70)},
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
                "output_embedding": "node",
                "output_mlp": {
                    "use_bias": [True, True, False], "units": [64, 32, 70],
                    "activation": ["relu", "relu", "softmax"]
                }
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "multi_target_indices": None,
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
                "loss": "categorical_crossentropy",
                "weighted_metrics": ["categorical_accuracy", {"class_name": "AUC", "config": {"name": "auc"}}]
            },
        },
        "dataset": {
            "class_name": "CoraDataset",
            "module_name": "kgcnn.data.datasets.CoraDataset",
            "config": {},
            "methods": [
                {"map_list": {"method": "make_undirected_edges"}},
                {"map_list": {"method": "add_edge_self_loops"}},
                {"map_list": {"method": "normalize_edge_weights_sym"}},
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
}
