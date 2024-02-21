# toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
trajectory_name = "ethanol_ccsd_t"

hyper = {
    "Schnet.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "Schnet",
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"}
                ],
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.Schnet",
                    "config": {
                        "name": "Schnet",
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
                                     "activation": [{"class_name": "function", "config": "kgcnn>shifted_softplus"},
                                                    {"class_name": "function", "config": "kgcnn>shifted_softplus"},
                                                    "linear"]},
                        "interaction_args": {
                            "units": 128, "use_bias": True,
                            "activation": {"class_name": "function", "config": "kgcnn>shifted_softplus"},
                            "cfconv_pool": "scatter_sum"
                        },
                        "node_pooling_args": {"pooling_method": "scatter_sum"},
                        "depth": 4,
                        "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                        "output_embedding": "graph",
                        "use_output_mlp": False,
                        "output_mlp": None
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3)}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 32}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss_weights": {"energy": 0.02, "force": 0.98}
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "dataset": {
            "class_name": "MD17Dataset",
            "module_name": "kgcnn.data.datasets.MD17Dataset",
            "config": {
                "trajectory_name": trajectory_name
            },
            "methods": [
                {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "R", "new_property_name": "node_coordinates"}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "atomic_number",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "postfix_file": "_" + trajectory_name,
            "kgcnn_version": "4.0.0"
        }
    },
    "PAiNN.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "PAiNN",
                "nested_model_config": True,
                "output_to_tensor": False,
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
                    "module_name": "kgcnn.literature.PAiNN",
                    "config": {
                        "name": "PAiNNEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                            {"shape": (), "name": "total_nodes", "dtype": "int64"},
                            {"shape": (), "name": "total_ranges", "dtype": "int64"}
                        ],
                        "input_embedding": None,
                        "cast_disjoint_kwargs": {"padded_disjoint": False},
                        "input_node_embedding": {"input_dim": 95, "output_dim": 128},
                        "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
                        "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                        "pooling_args": {"pooling_method": "scatter_sum"},
                        "conv_args": {"units": 128, "cutoff": None},
                        "update_args": {"units": 128}, "depth": 3, "verbose": 10,
                        "output_embedding": "graph",
                        "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3)}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Adam", "config": {
                        "learning_rate": {
                            "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                "learning_rate": 0.001, "warmup_steps": 150.0, "decay_steps": 20000.0,
                                "decay_rate": 0.01
                            }
                        }, "amsgrad": True, "use_ema": True
                    }
                },
                "loss_weights": {"energy": 0.02, "force": 0.98}
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
        },
        "dataset": {
            "class_name": "MD17Dataset",
            "module_name": "kgcnn.data.datasets.MD17Dataset",
            "config": {
                # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                "trajectory_name": trajectory_name
            },
            "methods": [
                {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "R", "new_property_name": "node_coordinates"}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "atomic_number",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "info": {
            "postfix": "",
            "postfix_file": "_" + trajectory_name,
            "kgcnn_version": "4.0.0"
        }
    },
    # Ragged!
    # The Metrics deviate from padded sum.
    "EGNN.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "EGNN",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": (None, 15), "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": (None, 1), "name": "range_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "model_energy": {
                    "module_name": "kgcnn.literature.EGNN",
                    "class_name": "make_model",
                    "config": {
                        "name": "EGNNEnergy",
                        "inputs": [
                            {"shape": (None, 15), "name": "node_attributes", "dtype": "float32", "ragged": True},
                            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                            {"shape": (None, 1), "name": "range_attributes", "dtype": "float32", "ragged": True},
                            {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                        ],
                        "input_tensor_type": "ragged",
                        "input_node_embedding": {"input_dim": 95, "output_dim": 128},
                        "input_edge_embedding": {"input_dim": 95, "output_dim": 128},
                        "depth": 7,
                        "node_mlp_initialize": {"units": 128, "activation": "linear"},
                        "euclidean_norm_kwargs": {"keepdims": True, "axis": 1, "square_norm": True},
                        "use_edge_attributes": False,
                        "edge_mlp_kwargs": {"units": [128, 128], "activation": ["swish", "swish"]},
                        "edge_attention_kwargs": {"units": 1, "activation": "sigmoid"},
                        "use_normalized_difference": False,
                        "expand_distance_kwargs": {"dim_half": 64},
                        "coord_mlp_kwargs": None,
                        # {"units": [128, 1], "activation": ["swish", "linear"]} or "tanh" at the end
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
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3), "ragged": True}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 96, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>CosineAnnealingLRScheduler", "config": {
                        "lr_start": 0.5e-03, "lr_min": 0.0, "epoch_max": 1000, "verbose": 1}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss_weights": {"energy": 0.02, "force": 0.98},
                "loss": {
                    "energy": "mean_absolute_error",
                    "force": {"class_name": "kgcnn>RaggedValuesMeanAbsoluteError", "config": {}}
                }
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "data": {
            "data_unit": "kcal/mol"
        },
        "dataset": {
            "class_name": "MD17Dataset",
            "module_name": "kgcnn.data.datasets.MD17Dataset",
            "config": {
                # toluene_ccsd_t, aspirin_ccsd, malonaldehyde_ccsd_t, benzene_ccsd_t, ethanol_ccsd_t
                "trajectory_name": trajectory_name
            },
            "methods": [
                {"rename_property_on_graphs": {"old_property_name": "E", "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "F", "new_property_name": "force"}},
                {"rename_property_on_graphs": {"old_property_name": "z", "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "R", "new_property_name": "node_coordinates"}},
                {"map_list": {"method": "atomic_charge_representation", "node_number": "atomic_number"}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "node_attributes",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "info": {
            "postfix": "",
            "postfix_file": "_" + trajectory_name,
            "kgcnn_version": "4.0.1"
        }
    },
}
