hyper = {
    "Schnet.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "Schnet",
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.Schnet",
                    "config": {
                        "name": "SchnetEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                            {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                        ],
                        "input_embedding": {
                            "node": {"input_dim": 95, "output_dim": 128}
                        },
                        "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                                     "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                        "interaction_args": {
                            "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
                        },
                        "node_pooling_args": {"pooling_method": "sum"},
                        "depth": 6,
                        "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                        "output_embedding": "graph",
                        "use_output_mlp": False,
                        "output_mlp": None,
                    }
                }
            }
        },
        "training": {
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 32}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "aspirin_ccsd"
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                                  "node_coordinates": "R"}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_aspirin_ccsd",
            "kgcnn_version": "3.1.0"
        }
    },
    "PAiNN.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "PAiNN",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.PAiNN",
                    "config": {
                        "name": "PAiNNEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                            {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                        ],
                        "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
                        "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
                        "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                        "pooling_args": {"pooling_method": "sum"}, "conv_args": {"units": 128, "cutoff": None},
                        "update_args": {"units": 128}, "depth": 3, "verbose": 10,
                        "output_embedding": "graph",
                        "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
                    }
                }
            }
        },
        "training": {
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 150.0, "decay_steps": 20000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                },
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "aspirin_ccsd"
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                                  "node_coordinates": "R"}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_aspirin_ccsd",
            "kgcnn_version": "3.1.0"
        }
    },
    "DimeNetPP.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "DimeNetPP",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [{"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                           {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True}],
                "model_energy": {
                    "module_name": "kgcnn.literature.DimeNetPP",
                    "class_name": "make_model",
                    "config": {
                        "name": "DimeNetPPEnergy",
                        "inputs": [{"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                                   {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                                   {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                                   {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True}],
                        "input_embedding": {"node": {"input_dim": 95, "output_dim": 128,
                                                     "embeddings_initializer": {
                                                         "class_name": "RandomUniform",
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
                }
            }
        },
        "training": {
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 10, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 150.0, "decay_steps": 20000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                },
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "aspirin_ccsd"
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 1000,
                                  "node_coordinates": "R"}},
                    {"map_list": {"method": "set_angle", "node_coordinates": "R"}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "_aspirin_ccsd",
            "kgcnn_version": "3.1.0"
        }
    },
    "NMPN.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "NMPN",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [{"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                           {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}],
                "model_energy": {
                    "module_name": "kgcnn.literature.NMPN",
                    "class_name": "make_model",
                    "config": {
                        "name": "NMPNEnergy",
                        "inputs": [{"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                                   {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                                   {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}],
                        "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                            "edge": {"input_dim": 95, "output_dim": 64}},
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
                }
            }
        },
        "training": {
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 32}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "aspirin_ccsd"
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 1000,
                                  "node_coordinates": "R"}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "_aspirin_ccsd",
            "kgcnn_version": "3.1.0"
        }
    },
    "Megnet.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "Megnet",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                    {"shape": [3], "name": "graph_inertia", "dtype": "float32", "ragged": False}
                ],
                "model_energy": {
                    "module_name": "kgcnn.literature.Megnet",
                    "class_name": "make_model",
                    "config": {
                        "name": "MegnetEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                            {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                            {"shape": [3], "name": "graph_inertia", "dtype": "float32", "ragged": False}
                        ],
                        "input_embedding": {"node": {"input_dim": 100, "output_dim": 16},
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
                }
            }
        },
        "training": {
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 32}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "aspirin_ccsd"
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                    {"map_list": {"method": "principal_moments_of_inertia", "node_mass": "z", "node_coordinates": "R"}},
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 1000,
                                  "node_coordinates": "R"}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "_aspirin_ccsd",
            "kgcnn_version": "3.1.0"
        }
    },
    "MXMNet.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "MXMNet",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [{"shape": (None,), "name": "atomic_number", "dtype": "float32", "ragged": True},
                           {"shape": (None, 3), "name": "R", "dtype": "float32", "ragged": True},
                           {"shape": (None, 1), "name": "edge_weights", "dtype": "float32", "ragged": True},
                           {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices_1", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices_2", "dtype": "int64", "ragged": True},
                           {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True}],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.MXMNet",
                    "config": {
                        "name": "MXMNetEnergy",
                        "inputs": [{"shape": (None,), "name": "atomic_number", "dtype": "float32", "ragged": True},
                                   {"shape": (None, 3), "name": "R", "dtype": "float32", "ragged": True},
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
                        "bessel_basis_global": {"num_radial": 16, "cutoff": 5.0, "envelope_exponent": 5},
                        "spherical_basis_local": {"num_spherical": 7, "num_radial": 6, "cutoff": 5.0,
                                                  "envelope_exponent": 5},
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
                }
            }
        },
        "training": {
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 128, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 32}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03, "global_clipnorm": 1000}},
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "aspirin_ccsd"
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                    # we have to make edges via range cutoff.
                    {"map_list": {"method": "set_range", "max_distance": 2.0, "max_neighbours": 1000,
                                  "node_coordinates": "R", "range_indices": "edge_indices",
                                  "range_attributes": "edge_distance"}},
                    {"map_list": {"method": "set_edge_weights_uniform"}},
                    {"map_list": {"method": "set_range", "max_distance": 5.0, "max_neighbours": 1000,
                                  "node_coordinates": "R"}},
                    {"map_list": {"method": "set_angle", "range_indices": "edge_indices", "edge_pairing": "jk",
                                  "angle_indices": "angle_indices_1", "node_coordinates": "R",
                                  "angle_indices_nodes": "angle_indices_nodes_1",
                                  "angle_attributes": "angle_attributes_1"}},
                    {"map_list": {"method": "set_angle", "range_indices": "edge_indices", "edge_pairing": "ik",
                                  "allow_self_edges": True,
                                  "angle_indices": "angle_indices_2", "node_coordinates": "R",
                                  "angle_indices_nodes": "angle_indices_nodes_2",
                                  "angle_attributes": "angle_attributes_2"}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "_aspirin_ccsd",
            "kgcnn_version": "3.1.0"
        }
    },
    "EGNN.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "EGNN",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [{"shape": (None, 15), "name": "node_attributes", "dtype": "float32", "ragged": True},
                           {"shape": (None, 3), "name": "R", "dtype": "float32", "ragged": True},
                           {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                           {"shape": (None, 1), "name": "range_attributes", "dtype": "int64", "ragged": True}],
                "model_energy": {
                    "module_name": "kgcnn.literature.EGNN",
                    "class_name": "make_model",
                    "config": {
                        "name": "EGNNEnergy",
                        "inputs": [{"shape": (None, 15), "name": "node_attributes", "dtype": "float32", "ragged": True},
                               {"shape": (None, 3), "name": "R", "dtype": "float32", "ragged": True},
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
                        "expand_distance_kwargs": {"dim_half": 64},
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
                }
            }
        },
        "training": {
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 96, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>CosineAnnealingLRScheduler", "config": {
                        "lr_start": 0.5e-03, "lr_min": 0.0, "epoch_max": 1000, "verbose": 1}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "aspirin_ccsd"
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                    {"map_list": {"method": "atomic_charge_representation", "node_number": "z"}},
                    {"map_list": {"method": "set_range", "max_distance": 10, "max_neighbours": 10000,
                                  "node_coordinates": "R"}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "_aspirin_ccsd",
            "kgcnn_version": "3.1.0"
        }
    },
}
