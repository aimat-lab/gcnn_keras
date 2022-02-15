# Training

Example training to test the performance of model implementations of ``kgcnn`` per datasets. The training scripts are called via:

```bash
python3 train_citation.py --dataset CoraLuDataset --model GAT --hyper hyper/hyper_esol.py
python3 train_moleculenet.py --dataset ESOLDataset --model GIN --hyper hyper/hyper_esol.py
python3 train_tudataset.py --dataset PROTEINSDataset --model GIN --hyper hyper/hyper_proteins.py
```

Where `hyper_esol.py` stores hyperparameter and must be in the same folder or a path to a `.py`. 
Alternatively, also a `.yaml`, `.yaml` or `.json` file can be loaded in place of the `.py` file. 
The python file must define a `hyper` attribute as described below.
However, note that not all models can be trained on all datasets and that not all models have proper default hyperparameter here.
You can check previous runs in the result folders named after each dataset and their hyperparameter and output files.

There is a [``make_config_training.ipynb``](/notebooks/tutorial_config_training.ipynb) jupyter [notebook](/notebooks) to demonstrate how the `.py` config file can be set up and provide further information
on how it is structured. In short the config file contains a python dictionary of the form:

```python3
hyper = {
    "info":{ 
        # General information for training run
        "kgcnn_version": "2.0.0", # Version 
        "postfix": "" # Postfix for output folder.
    },
    "model": { 
        # Model specific parameter, see kgcnn.literature.
    },
    "data": { 
        # Dataset specific parameters.
    },
    "training": {
        "fit": { 
            # serialized keras fit arguments.
        },
        "compile": { 
            # serialized keras compile arguments.
        },
        "cross_validation": {
            # serialized parameters for cross-validation.  
        },
        "scaler": {
            # serialized parameters for scaler.
            # Only add when training for regression.
        }
    }
}
```

Furthermore, you could also have a dictionary of models as ``hyper={"GCN": {...}, ...}`` which each model has a config dictionary as above.

If a python file is used, also non-serialized hyperparameter for fit and compile can be provided. 
However, note that the python file will be executed, and a serialization after model fit may fail depending on the arguments.