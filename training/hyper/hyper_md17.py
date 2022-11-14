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
                "config": {
                    "name": "SchnetEnergy",
                    "inputs": [
                        {"shape": [None], "name": "z", "dtype": "float32", "ragged": True},
                        {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
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
            "train_test_indices": {},
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 1000,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}}
            },
            "scaler": {"class_name": "EnergyForceExtensiveScaler",
                       "config": {}},
            "multi_target_indices": 1,  # Number of states
        },
        "data": {
            "dataset": {
                "class_name": "MD17Dataset",
                "module_name": "kgcnn.data.datasets.MD17Dataset",
                "config": {"trajectory_name": "aspirin_ccsd"},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                                  "node_coordinates": "R"}}
                ]
            },
        },
        "info": {
            "postfix": "",
            "postfix_file": "_aspirin_ccsd",
            "kgcnn_version": "2.2.0"
        }
    },
}
