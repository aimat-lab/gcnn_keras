# toluene, aspirin, malonaldehyde, benzene, ethanol
trajectory_name = "toluene"

hyper = {
    "Schnet.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "Schnet",
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
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 32}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss_weights": {"energy": 1.0, "force": 49.0}
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MD17RevisedDataset",
                "module_name": "kgcnn.data.datasets.MD17RevisedDataset",
                "config": {
                    # toluene, aspirin, malonaldehyde, benzene, ethanol
                    "trajectory_name": trajectory_name
                },
                "methods": [
                    {"rename_property_on_graphs": {"old_property_name": "energies", "new_property_name": "energy"}},
                    {"rename_property_on_graphs": {"old_property_name": "forces", "new_property_name": "force"}},
                    {"rename_property_on_graphs": {"old_property_name": "nuclear_charges",
                                                   "new_property_name": "atomic_number"}},
                    {"rename_property_on_graphs": {"old_property_name": "coords",
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
            "postfix_file": "_"+trajectory_name,
            "kgcnn_version": "4.0.0"
        }
    },
}