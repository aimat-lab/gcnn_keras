hyper = {
    "Schnet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.Schnet",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64}
                },
                "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                             "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
                },
                "node_pooling_args": {"pooling_method": "sum"},
                "depth": 4,
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": None,
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
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
            },
            "scaler": {"class_name": "QMGraphLabelScaler", "config": {
                "scaler": [{"class_name": "StandardScaler",
                            "config": {"with_std": True, "with_mean": True, "copy": True}},
                           ]
            }},
            "multi_target_indices": [5]  # 5, 6, 7 = Homo, Lumo, Gap or combination
        },
        "data": {
            "dataset": {
                "class_name": "QM9Dataset",
                "module_name": "kgcnn.data.datasets.QM9Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 30}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_HOMO",
            "kgcnn_version": "2.1.0"
        }
    },
    "Megnet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.Megnet",
            "config": {
                "name": "Megnet",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                    {"shape": [2], "name": "graph_attributes", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 10, "output_dim": 16},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
                "meg_block_args": {"node_embed": [64, 32, 32], "edge_embed": [64, 32, 32],
                                   "env_embed": [64, 32, 32], "activation": "kgcnn>softplus2"},
                "set2set_args": {"channels": 16, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
                "node_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
                "edge_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
                "state_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
                "nblocks": 3, "has_ff": True, "dropout": None, "use_set2set": True,
                "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                               "activation": ["kgcnn>softplus2", "kgcnn>softplus2", "linear"]},
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 32, "epochs": 800, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 800,
                        "verbose": 0
                    }
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
                "loss": "mean_absolute_error"
            },
            "scaler": {"class_name": "QMGraphLabelScaler", "config": {
                "scaler": [{"class_name": "StandardScaler",
                            "config": {"with_std": True, "with_mean": True, "copy": True}}
                           ]
            }},
            "multi_target_indices": [5]  # 5, 6, 7 = Homo, Lumo, Gap or combination
        },
        "data": {
            "dataset": {
                "class_name": "QM9Dataset",
                "module_name": "kgcnn.data.datasets.QM9Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 30}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_HOMO",
            "kgcnn_version": "2.1.0"
        }
    },
    "NMPN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.NMPN",
            "config": {
                "name": "NMPN",
                "inputs": [{"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                           {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64}},
                "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
                "pooling_args": {"pooling_method": "segment_sum"},
                "use_set2set": True,
                "depth": 3,
                "node_dim": 128,
                "verbose": 10,
                "geometric_edge": True, "make_distance": True, "expand_distance": True,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [25, 25, 1],
                               "activation": ["selu", "selu", "linear"]},
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 32, "epochs": 700, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-04, "learning_rate_stop": 1e-05, "epo_min": 50, "epo": 700,
                        "verbose": 0
                    }
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-04}},
                "loss": "mean_absolute_error"
            },
            "scaler": {"class_name": "QMGraphLabelScaler", "config": {
                "scaler": [{"class_name": "StandardScaler",
                            "config": {"with_std": True, "with_mean": True, "copy": True}}
                           ]
            }},
            "multi_target_indices": [5]  # 5, 6, 7 = Homo, Lumo, Gap or combination
        },
        "data": {
            "dataset": {
                "class_name": "QM9Dataset",
                "module_name": "kgcnn.data.datasets.QM9Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 30}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_HOMO",
            "kgcnn_version": "2.1.0"
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
            "multi_target_indices": [5]  # 5, 6, 7 = Homo, Lumo, Gap or combination
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
            "postfix_file": "_HOMO",
            "kgcnn_version": "2.1.0"
        }
    },
    "DimeNetPP": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DimeNetPP",
            "config": {
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
                "num_targets": 1, "extensive": False, "output_init": "zeros",
                "activation": "swish", "verbose": 10,
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": {},
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 10, "epochs": 600, "validation_freq": 10, "verbose": 2, "callbacks": []
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
            "multi_target_indices": [5]  # 5, 6, 7 = Homo, Lumo, Gap or combination
        },
        "data": {
            "dataset": {
                "class_name": "QM9Dataset",
                "module_name": "kgcnn.data.datasets.QM9Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 1000}},
                    {"map_list": {"method": "set_angle"}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_HOMO",
            "kgcnn_version": "2.1.0"
        }
    },
    "MXMNet": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MXMNet",
            "config": {
                "name": "MXMNet",
                "inputs": [{"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
                           {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": (None, 1), "name": "edge_weights", "dtype": "float32", "ragged": True},
                           {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices_1", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices_2", "dtype": "int64", "ragged": True},
                           {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 128, "embeddings_initializer": {
                                "class_name": "RandomUniform",
                                "config": {"minval": -1.7320508075688772, "maxval": 1.7320508075688772}}},
                    "edge": {"input_dim": 32, "output_dim": 128}},
                "bessel_basis_local": {"num_radial": 16, "cutoff": 5.0, "envelope_exponent": 5},
                "bessel_basis_global": {"num_radial": 16, "cutoff": 10.0, "envelope_exponent": 5},
                "spherical_basis_local": {"num_spherical": 7, "num_radial": 6, "cutoff": 5.0, "envelope_exponent": 5},
                "mlp_rbf_kwargs": {"units": 128, "activation": "swish"},
                "mlp_sbf_kwargs": {"units": 128, "activation": "swish"},
                "global_mp_kwargs": {"units": 128},
                "local_mp_kwargs": {"units": 128, "output_units": 1,
                                    "output_kernel_initializer": "glorot_uniform"},
                "use_edge_attributes": False,
                "depth": 6,
                "verbose": 10,
                "node_pooling_args": {"pooling_method": "sum"},
                "output_embedding": "graph", "output_to_tensor": True,
                "use_output_mlp": False,
                "output_mlp": {"use_bias": [True], "units": [1],
                               "activation": ["linear"]}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 128, "epochs": 900, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.9961697, "epo_warmup": 1, "verbose": 1}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03, "global_clipnorm": 1000}},
                "loss": "mean_absolute_error",
                "metrics": [
                    "mean_absolute_error", "mean_squared_error",
                    # No scaling needed.
                    {"class_name": "RootMeanSquaredError", "config": {"name": "scaled_root_mean_squared_error"}},
                    {"class_name": "MeanAbsoluteError", "config": {"name": "scaled_mean_absolute_error"}},
                ]
            },
            "multi_target_indices": [6]  # 5, 6, 7 = Homo, Lumo, Gap or combination
        },
        "data": {
            "dataset": {
                "class_name": "QM9Dataset",
                "module_name": "kgcnn.data.datasets.QM9Dataset",
                "config": {},
                "methods": [
                    {"remove_uncharacterized": {}},
                    {"map_list": {"method": "set_edge_weights_uniform"}},
                    {"map_list": {"method": "set_range", "max_distance": 10, "max_neighbours": 1000}},
                    {"map_list": {"method": "set_angle", "range_indices": "edge_indices", "edge_pairing": "jk",
                                  "angle_indices": "angle_indices_1",
                                  "angle_indices_nodes": "angle_indices_nodes_1",
                                  "angle_attributes": "angle_attributes_1"}},
                    {"map_list": {"method": "set_angle", "range_indices": "edge_indices", "edge_pairing": "ik",
                                  "allow_self_edges": True,
                                  "angle_indices": "angle_indices_2",
                                  "angle_indices_nodes": "angle_indices_nodes_2",
                                  "angle_attributes": "angle_attributes_2"}}
                ]
            },
            "data_unit": "eV"
        },
        "info": {
            "postfix": "",
            "postfix_file": "_LUMO",
            "kgcnn_version": "2.1.1"
        }
    },
    "EGNN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.EGNN",
            "config": {
                "name": "EGNN",
                "inputs": [{"shape": (None, 15), "name": "node_attributes", "dtype": "float32", "ragged": True},
                           {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                           {"shape": (None, 1), "name": "range_attributes", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 95, "output_dim": 128}},
                "depth": 7,
                "node_mlp_initialize": {"units": 128, "activation": "linear"},
                "euclidean_norm_kwargs": {"keepdims": True, "axis": 2, "square_norm": True},
                "use_edge_attributes": False,
                "edge_mlp_kwargs": {"units": [128, 128], "activation": ["swish", "swish"]},
                "edge_attention_kwargs": {"units": 1, "activation": "sigmoid"},
                "use_normalized_difference": False,
                "expand_distance_kwargs": None,
                "coord_mlp_kwargs": None,  # {"units": [128, 1], "activation": ["swish", "linear"]} or "tanh" at the end
                "pooling_coord_kwargs": None,  # {"pooling_method": "mean"},
                "pooling_edge_kwargs": {"pooling_method": "sum"},
                "node_normalize_kwargs": None,
                "use_node_attributes": False,
                "node_mlp_kwargs": {"units": [128, 128], "activation": ["swish", "linear"]},
                "use_skip": True,
                "verbose": 10,
                "node_decoder_kwargs": {"units": [128, 128], "activation": ["swish", "linear"]},
                "node_pooling_kwargs": {"pooling_method": "sum"},
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True], "units": [128, 1],
                               "activation": ["swish", "linear"]}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 10, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 96, "epochs": 800, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>CosineAnnealingLRScheduler", "config": {
                        "lr_start": 0.5e-03, "lr_min": 0.0, "epoch_max": 800, "verbose": 1}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.5e-03}},
                "loss": "mean_absolute_error"
            },
            "scaler": {"class_name": "QMGraphLabelScaler", "config": {
                "scaler": [
                    {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}]
                }
            },
            "multi_target_indices": [6]  # 5, 6, 7 = Homo, Lumo, Gap or combination
        },
        "data": {
            "dataset": {
                "class_name": "QM9Dataset",
                "module_name": "kgcnn.data.datasets.QM9Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "atomic_charge_representation"}},
                    {"map_list": {"method": "set_range", "max_distance": 10, "max_neighbours": 10000}}
                ]
            },
            "data_unit": "eV"
        },
        "info": {
            "postfix": "",
            "postfix_file": "_LUMO",
            "kgcnn_version": "2.1.1"
        }
    },
}
