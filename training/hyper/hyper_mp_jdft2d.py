hyper = {
    "Megnet.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.Megnet",
            "class_name": "make_crystal_model",
            "config": {
                'name': "Megnet",
                'inputs': [{'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                           {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                           {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}],
                'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "make_distance": True, "expand_distance": True,
                'gauss_args': {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4},
                'meg_block_args': {'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32],
                                   'env_embed': [64, 32, 32], 'activation': 'kgcnn>softplus2'},
                'set2set_args': {'channels': 16, 'T': 3, "pooling_method": "sum", "init_qstar": "0"},
                'node_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                'edge_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                'state_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                'nblocks': 3, 'has_ff': True, 'dropout': None, 'use_set2set': True,
                'verbose': 10,
                'output_embedding': 'graph',
                'output_mlp': {"use_bias": [True, True, True], "units": [32, 16, 1],
                               "activation": ['kgcnn>softplus2', 'kgcnn>softplus2', 'linear']}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.0005, "learning_rate_stop": 0.5e-05, "epo_min": 100, "epo": 1000,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0}}
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },
    "Schnet.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.Schnet",
            "class_name": "make_crystal_model",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                    {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                    {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 64}
                },
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
                },
                "node_pooling_args": {"pooling_method": "mean"},
                "depth": 4,
                "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                             "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": None,  # Last MLP sets output dimension if None.
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
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
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5}}
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },
    "PAiNN.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.PAiNN",
            "class_name": "make_crystal_model",
            "config": {
                "name": "PAiNN",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
                "equiv_initialize_kwargs": {"dim": 3, "method": "eye"},
                "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                "pooling_args": {"pooling_method": "mean"},
                "conv_args": {"units": 128, "cutoff": None, "conv_pool": "sum"},
                "update_args": {"units": 128}, "depth": 2, "verbose": 10,
                "equiv_normalization": False, "node_normalization": False,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]}
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 32, "epochs": 800, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupLinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-04, "learning_rate_stop": 1e-06, "epo_warmup": 25, "epo": 1000,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-04}},  # "clipnorm": 100.0, "clipvalue": 100.0}
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0}}
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },
    "DimeNetPP.make_crystal_model": {
        "model": {
            "class_name": "make_crystal_model",
            "module_name": "kgcnn.literature.DimeNetPP",
            "config": {
                "name": "DimeNetPP",
                "inputs": [{"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                           {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
                           {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                           {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                           ],
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
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 16, "epochs": 780, "validation_freq": 10, "verbose": 2, "callbacks": [],
                "validation_batch_size": 8
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
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 17}},
                    {"map_list": {"method": "set_angle", "allow_multi_edges": True, "allow_reverse_edges": True}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },
    "CGCNN.make_crystal_model": {
        "model": {
            "class_name": "make_crystal_model",
            "module_name": "kgcnn.literature.CGCNN",
            "config": {
                'name': 'CGCNN',
                'inputs': [
                    {'shape': (None,), 'name': 'node_number', 'dtype': 'int64', 'ragged': True},
                    {'shape': (None, 3), 'name': 'node_frac_coordinates', 'dtype': 'float64', 'ragged': True},
                    {'shape': (None, 2), 'name': 'range_indices', 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': 'graph_lattice', 'dtype': 'float64', 'ragged': False},
                    {'shape': (None, 3), 'name': 'range_image', 'dtype': 'float32', 'ragged': True},
                    # For `representation="asu"`:
                    # {'shape': (None, 1), 'name': 'multiplicities', 'dtype': 'float32', 'ragged': True},
                    # {'shape': (None, 4, 4), 'name': 'symmops', 'dtype': 'float64', 'ragged': True},
                ],
                'input_embedding': {'node': {'input_dim': 95, 'output_dim': 64}},
                'representation': 'unit',  # None, 'asu' or 'unit'
                'expand_distance': True,
                'make_distances': True,
                'gauss_args': {'bins': 60, 'distance': 6, 'offset': 0.0, 'sigma': 0.4},
                'conv_layer_args': {
                    'units': 128,
                    'activation_s': 'kgcnn>shifted_softplus',
                    'activation_out': 'kgcnn>shifted_softplus',
                    'batch_normalization': True,
                },
                'node_pooling_args': {'pooling_method': 'mean'},
                'depth': 4,
                'output_mlp': {'use_bias': [True, True, False], 'units': [128, 64, 1],
                               'activation': ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 128, "epochs": 1000, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-05, "epo_min": 500, "epo": 1000,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 6.0}}
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },
    "NMPN.make_crystal_model": {
        "model": {
            "class_name": "make_crystal_model",
            "module_name": "kgcnn.literature.NMPN",
            "config": {
                "name": "NMPN",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64}},
                "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
                "pooling_args": {"pooling_method": "segment_mean"},
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
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 16, "epochs": 700, "validation_freq": 10, "verbose": 2,
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
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },
    "MEGAN": {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MEGAN",
            "config": {
                'name': "MEGAN",
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64, "use_embedding": True}},
                'units': [60, 50, 40, 30],
                'importance_units': [],
                'final_units': [50, 30, 10, 1],
                "final_activation": "linear",
                'dropout_rate': 0.3,
                'final_dropout_rate': 0.00,
                'importance_channels': 3,
                'return_importances': False,
                'use_edge_features': True,
                'inputs': [{'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 25), 'name': "range_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True}],
            }
        },
        "training": {
            "fit": {
                "batch_size": 64,
                "epochs": 400,
                "validation_freq": 1,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-05, "epo_min": 50, "epo": 400,
                        "verbose": 0}
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0, "max_neighbours": 40}},
                    {"map_list": {"method": "expand_distance_gaussian_basis", "distance": 5.0, "bins": 25,
                                  "expand_dims": False}}
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.3"
        }
    },
    "coGN": {
        "model": {
            "module_name": "kgcnn.literature.coGN",
            "class_name": "make_model",
            "config": {
                "name": "coGN",
                "inputs": {
                    "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                    "cell_translation": None,
                    "affine_matrix": None,
                    "voronoi_ridge_area": None,
                    "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                    "frac_coords": None,
                    "coords": None,
                    "multiplicity": {"shape": (None,), "name": "multiplicity", "dtype": "int32", "ragged": True},
                    "lattice_matrix": None,
                    "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int32", "ragged": True},
                    "line_graph_edge_indices": None,
                },
                # All default.
            }
        },
        "training": {
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.0005, "learning_rate_stop": 0.5e-05, "epo_min": 100, "epo": 1000,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectJdft2dDataset",
                "module_name": "kgcnn.data.datasets.MatProjectJdft2dDataset",
                "config": {},
                "methods": [
                    {"set_representation": {
                        "pre_processor": {"class_name": "KNNAsymmetricUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"k": 24}
                                          },
                        "reset_graphs": False}}
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.1"
        }
    },
}
