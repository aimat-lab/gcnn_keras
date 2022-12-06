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
                'units': [2, 2, 2],
                'importance_units': [],
                'final_units': [2],
                "final_activation": "softmax",
                'dropout_rate': 0.0,
                'importance_factor': 1.0,
                'importance_multiplier': 1.0,
                'sparsity_factor': 2.0,
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
}