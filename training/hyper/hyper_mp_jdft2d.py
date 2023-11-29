hyper = {
    "Schnet.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.Schnet",
            "class_name": "make_crystal_model",
            "config": {
                "name": "Schnet",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "int32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64'},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32'},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"}
                ],
                "cast_disjoint_kwargs": {"padded_disjoint": False},
                "input_node_embedding": {"input_dim": 95, "output_dim": 64},
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus",
                    "cfconv_pool": "scatter_sum"
                },
                "node_pooling_args": {"pooling_method": "scatter_mean"},
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
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 0.0005}},
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardLabelScaler",
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
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                  "count_edges": "range_indices", "count_nodes": "node_number",
                                  "total_nodes": "total_nodes"}},
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
        }
    },
    "PAiNN.make_crystal_model": {
        "model": {
            "module_name": "kgcnn.literature.PAiNN",
            "class_name": "make_crystal_model",
            "config": {
                "name": "PAiNN",
                "inputs": [
                    {"shape": [None], "name": "node_number", "dtype": "int64", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                    {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                    {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                ],
                "input_tensor_type": "ragged",
                "cast_disjoint_kwargs": {},
                "input_node_embedding": {"input_dim": 95, "output_dim": 128},
                "equiv_initialize_kwargs": {"dim": 3, "method": "eye"},
                "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                "pooling_args": {"pooling_method": "scatter_mean"},
                "conv_args": {"units": 128, "cutoff": None, "conv_pool": "scatter_sum"},
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
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-04}},  # "clipnorm": 100.0, "clipvalue": 100.0}
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardLabelScaler",
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
                    {"map_list": {"method": "set_range_periodic", "max_distance": 5.0}},
                    {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                                  "count_edges": "range_indices", "count_nodes": "node_number",
                                  "total_nodes": "total_nodes"}},
                ]
            },
            "data_unit": "meV/atom"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.0"
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
                    {'shape': (None, 3), 'name': 'range_image', 'dtype': 'float32', 'ragged': True},
                    {'shape': (3, 3), 'name': 'graph_lattice', 'dtype': 'float64', 'ragged': False},
                    # For `representation="asu"`:
                    # {'shape': (None, 1), 'name': 'multiplicities', 'dtype': 'float32', 'ragged': True},
                    # {'shape': (None, 4, 4), 'name': 'symmops', 'dtype': 'float64', 'ragged': True},
                ],
                "input_tensor_type": "ragged",
                'input_node_embedding': {'input_dim': 95, 'output_dim': 64},
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
                'node_pooling_args': {'pooling_method': 'scatter_mean'},
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
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "scaler": {
                "class_name": "StandardLabelScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
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
            "kgcnn_version": "4.0.0"
        }
    },
}