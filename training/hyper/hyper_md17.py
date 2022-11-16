hyper = {
    "Schnet.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "Schnet",
                "class_name": "make_model",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "module_name": "kgcnn.literature.Schnet",
                "coordinate_input": 1,
                "config": {
                    "name": "SchnetEnergy",
                    "inputs": [
                        {"shape": [None], "name": "z", "dtype": "float32", "ragged": True},
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
        },
        "training": {
            "target_property_names": {
                "energy": "E", "force": "F", "atomic_number": "z",
                "coordinates": "R"},
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "ethanol_ccsd_t"
                },
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                                  "node_coordinates": "R"}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_ethanol_ccsd_t",
            "kgcnn_version": "2.2.0"
        }
    },
    "PAiNN.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.model.force",
            "config": {
                "name": "PAiNN",
                "class_name": "make_model",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "module_name": "kgcnn.literature.PAiNN",
                "coordinate_input": 1,
                "config": {
                    "name": "PAiNNEnergy",
                    "inputs": [
                        {"shape": [None], "name": "z", "dtype": "float32", "ragged": True},
                        {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                        {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                    ],
                    "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
                    "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
                    "bessel_basis": {"num_radial": 25, "cutoff": 5.0, "envelope_exponent": 5},
                    "pooling_args": {"pooling_method": "sum"}, "conv_args": {"units": 128, "cutoff": None},
                    "update_args": {"units": 128}, "depth": 6, "verbose": 10,
                    "output_embedding": "graph",
                    "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
                }
            }
        },
        "training": {
            "target_property_names": {
                "energy": "E", "force": "F", "atomic_number": "z",
                "coordinates": "R"},
            "train_test_indices": {"train": "train", "test": "test"},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss_weights": [1.0, 49.0]
            },
            "scaler": {"class_name": "EnergyForceExtensiveScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {
                    # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                    "trajectory_name": "ethanol_ccsd_t"
                },
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                                  "node_coordinates": "R"}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_ethanol_ccsd_t",
            "kgcnn_version": "2.2.0"
        }
    },
}
