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
            "cross_validation": None,
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
                            "config": {"with_std": True, "with_mean": True, "copy": True}}
                           ]
            }},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "QM7Dataset",
                "module_name": "kgcnn.data.datasets.QM7Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 30}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.1"
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
                "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
                "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                "pooling_args": {"pooling_method": "sum"}, "conv_args": {"units": 128, "cutoff": None},
                "update_args": {"units": 128}, "depth": 3, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
            }
        },
        "training": {
            "cross_validation": None,
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
                                        "learning_rate": 0.001, "warmup_steps": 150.0, "decay_steps": 200000.0,
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
                            "config": {"with_std": True, "with_mean": True, "copy": True}}
                           ]
            }},
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "QM7Dataset",
                "module_name": "kgcnn.data.datasets.QM7Dataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000}}
                ]
            },
            "data_unit": "kcal/mol"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.1"
        }
    },
}
