hyper = {
    "Schnet": {
        "model": {
            "module_name": "kgcnn.literature.Schnet",
            "class_name": "make_model",
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
                                 "config": {"n_splits": 10, "random_state": None, "shuffle": True}},
            "execute_folds": 1,
            "fit": {
                "batch_size": 64, "epochs": 300, "validation_freq": 10, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 300,
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
                "module_name": "kgcnn.scaler.scaler",
                "config": {"with_std": True, "with_mean": True, "copy": True}
            },
            "multi_target_indices": None
        },
        "data": {
            "dataset": {
                "class_name": "MatProjectEFormDataset",
                "module_name": "kgcnn.data.datasets.MatProjectEFormDataset",
                "config": {},
                "methods": [
                    # Does not take into account periodic structure!
                    {"map_list": {"method": "set_range", "max_distance": 10, "max_neighbours": 20}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    },
    # "PAiNN": {
    #     "model": {
    #         "module_name": "kgcnn.literature.PAiNN",
    #         "class_name": "make_model",
    #         "config": {
    #             "name": "PAiNN",
    #             "inputs": [
    #                 {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
    #                 {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
    #                 {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
    #             ],
    #             "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
    #             "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
    #             "pooling_args": {"pooling_method": "mean"},
    #             "conv_args": {"units": 128, "cutoff": None, "conv_pool": "mean"},
    #             "update_args": {"units": 128}, "depth": 2, "verbose": 10,
    #             "output_embedding": "graph",
    #             "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]}
    #         }
    #     },
    #     "training": {
    #         "cross_validation": {"class_name": "KFold",
    #                              "config": {"n_splits": 10, "random_state": None, "shuffle": True}},
    #         "execute_folds": 1,
    #         "fit": {
    #             "batch_size": 64, "epochs": 300, "validation_freq": 10, "verbose": 2,
    #             "callbacks": [
    #                 {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
    #                     "learning_rate_start": 0.0005, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 300,
    #                     "verbose": 0}
    #                  }
    #             ]
    #         },
    #         "compile": {
    #             "optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}},
    #             "loss": "mean_absolute_error"
    #         },
    #         "scaler": {
    #             "class_name": "StandardScaler",
    #             "module_name": "kgcnn.scaler.scaler",
    #             "config": {"with_std": True, "with_mean": True, "copy": True}
    #         },
    #         "multi_target_indices": None
    #     },
    #     "data": {
    #         "dataset": {
    #             "class_name": "MatProjectEFormDataset",
    #             "module_name": "kgcnn.data.datasets.MatProjectEFormDataset",
    #             "config": {},
    #             "methods": [
    #                 # Does not take into account periodic structure!
    #                 {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10}}
    #             ]
    #         },
    #     },
    #     "info": {
    #         "postfix": "",
    #         "postfix_file": "",
    #         "kgcnn_version": "2.0.3"
    #     }
    # }
}