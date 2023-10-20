hyper = {
    "Schnet.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "Schnet",
                "nested_model_config": True,
                "output_to_tensor": True,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"}
                ],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.Schnet",
                    "config": {
                        "name": "SchnetEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                            {"shape": (), "name": "total_nodes", "dtype": "int64"},
                            {"shape": (), "name": "total_ranges", "dtype": "int64"}
                        ],
                        "cast_disjoint_kwargs": {"padded_disjoint": False},
                        "input_node_embedding": {"input_dim": 95, "output_dim": 128},
                        "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                                     "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                        "interaction_args": {
                            "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus",
                            "cfconv_pool": "scatter_sum"
                        },
                        "node_pooling_args": {"pooling_method": "scatter_sum"},
                        "depth": 6,
                        "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                        "output_embedding": "graph",
                        "use_output_mlp": False,
                        "output_mlp": None,
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3)}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 64, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 8062}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss_weights": {"energy": 1.0, "force": 49.0}
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "data": {
            "dataset": {
                "class_name": "ISO17Dataset",
                "module_name": "kgcnn.data.datasets.ISO17Dataset",
                "config": {},
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "total_energy", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "atomic_forces", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "numbers",
                                                   "new_property_name": "atomic_number"}},
                    {"rename_property_on_graphs": {"old_property_name": "positions",
                                                   "new_property_name": "node_coordinates"}},
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                                  "node_coordinates": "node_coordinates"}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                  "count_edges": "range_indices", "count_nodes": "atomic_number",
                                  "total_nodes": "total_nodes"}},
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_test_within",
            "kgcnn_version": "4.0.0"
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
                    {"shape": [None, 3], "name": "positions", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "model_energy": {
                    "module_name": "kgcnn.literature.PAiNN",
                    "class_name": "make_model",
                    "config": {
                        "name": "PAiNNEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "float32", "ragged": True},
                            {"shape": [None, 3], "name": "positions", "dtype": "float32", "ragged": True},
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
            "train_test_indices": {"train": "train", "test": "test", "split_index": [0]},
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
                "class_name": "ISO17Dataset",
                "module_name": "kgcnn.data.datasets.ISO17Dataset",
                "config": {
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "total_energy", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "atomic_forces", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "numbers",
                                                   "new_property_name": "atomic_number"}},
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                                  "node_coordinates": "positions"}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_test_within",
            "kgcnn_version": "2.2.2"
        }
    },
}