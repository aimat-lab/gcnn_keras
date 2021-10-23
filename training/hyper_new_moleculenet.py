# Training new MoleculeNet
# Adjust file_name and data_directory

hyper = {
    "DMPNN": {
        "model": {
            "name": "DMPNN",
            "inputs": [
                {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                {"shape": [None, 1], "name": "edge_indices_reverse_pairs", "dtype": "int64", "ragged": True}
            ],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64},
                "edge": {"input_dim": 5, "output_dim": 64}
            },
            "output_embedding": "graph",
            "output_mlp": {
                "use_bias": [True, True, False], "units": [64, 32, 1],
                "activation": ["relu", "relu", "linear"]
            },
            "pooling_args": {"pooling_method": "sum"},
            "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
            "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
            "edge_activation": {"activation": "relu"},
            "node_dense": {"units": 128, "use_bias": True, "activation": "relu"},
            "verbose": 1, "depth": 5,
            "dropout": {"rate": 0.1}
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 5800,
                                             "decay_rate": 0.5, "staircase": False}}}
                              },
                "loss": "mean_absolute_error"
            },
            "KFold": {"n_splits": 5, "random_state": None, "shuffle": True},
            "execute_folds": None
        },
        "data": {
            # Adjust parameters for new dataset
            # Dataset must in a specific folder. E.g. named after Dataset.
            # See 'example_custom_moleculenet.ipynb' for more information
            "dataset": {"file_name": "example.csv", "data_directory": "Example/", "dataset_name": "Example"},
            "prepare_data": {"overwrite": True, "smiles_column_name": "smiles", "make_conformers": False},
            "read_in_memory": {"label_column_name": "labels", "add_hydrogen": False, "has_conformers": False},
            "set_attributes": {},  # Default arguments
            "set_edge_indices_reverse_pairs": {},
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "1.1.0"
        }
    }
}
