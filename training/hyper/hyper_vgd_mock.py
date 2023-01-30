hyper = {
    "MEGAN": {
        "explanation": {
            "channels": 2,
            "gt_suffix": None,
        },
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MEGAN",
            "config": {
                'name': "MEGAN",
                'units': [3, 3, 3],
                'importance_units': [],
                'final_units': [2],
                "final_activation": "softmax",
                'dropout_rate': 0.0,
                'importance_factor': 1.0,
                'importance_multiplier': 2.0,
                'sparsity_factor': 3.0,
                'final_dropout_rate': 0.00,
                'importance_channels': 2,
                'return_importances': False,
                'use_edge_features': False,
                'inputs': [{'shape': (None, 1), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 1), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
            }
        },
        "training": {
            "fit": {
                "batch_size": 8,
                "epochs": 100,
                "validation_freq": 1,
                "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-02}},
                "loss": "categorical_crossentropy",
                "metrics": ["categorical_accuracy"],
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "VgdMockDataset",
                "module_name": "kgcnn.data.datasets.VgdMockDataset",
                "config": {},
                "methods": []
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.0"
        }
    },
    "GCN": {
        "explanation": {
            "channels": 2,
            "gt_suffix": None,
        },
        "xai_methods": {
            "Mock": {
                "class_name": "MockImportanceExplanationMethod",
                "module_name": "kgcnn.xai.base",
                "config": {}
            },
            "GnnExplainer": {
                "class_name": "GnnExplainer",
                "module_name": "kgcnn.literature.GNNExplain",
                "config": {
                    "learning_rate": 0.01,
                    "epochs": 250,
                    "node_sparsity_factor": 0.1,
                    "edge_sparsity_factor": 0.1,
                }
            }
        },
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GCN",
            "config": {
                "name": "GCN",
                'inputs': [{'shape': (None, 1), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 1), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                "gcn_args": {"units": 32, "use_bias": True, "activation": "relu"},
                "depth": 3, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [32, 16, 2],
                               "activation": ["relu", "relu", "softmax"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 100,
                "validation_freq": 10,
                "verbose": 2,
            },
            "compile": {
                "optimizer": {"class_name": "Nadam", "config": {"lr": 1e-02}},
                "loss": "categorical_crossentropy",
                "metrics": ["categorical_accuracy"],
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
        },
        "data": {
            "dataset": {
                "class_name": "VgdMockDataset",
                "module_name": "kgcnn.data.datasets.VgdMockDataset",
                "config": {},
                "methods": []
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    }
}