# Training

Example training to test the performance of model implementations of ``kgcnn`` per datasets. The training scripts are called via:

```bash
python3 train_qm9.py --dataset QM9
python3 train_qm9.py --model Schnet
python3 train_qm9.py --model Schnet --hpyer config.json
```

Where `config.json` stores custom hyper-parameters and must be in the same folder or a path to a `.json` file. 
Alternatively, also a `.yaml`, `.yaml` or `.py` file can be loaded in place of the `.json` file. 
The python file must define a ```hyper``` attribute as described below.
However, note that not all models can be trained on all datasets and that not all models have proper default hyper parameters here.
You can check previous runs in the result folders named after each dataset and their hyper parameters and output files.

There is a [``make_config_training.ipynb``](/notebooks/make_config_training.ipynb) jupyter [notebook](/notebooks) to demonstrate how the `.py` config file can be set up and provide further information
on how it is structured. In short the config file contains a python dictionary of the form:

```python3
hyper = {
    "info":{ 
        # General information for training run
        "kgcnn_version": "1.1.0", # Version 
        "postfix": "" # postfix for output folder
    },
    "model": { 
        # Model specific parameter, see kgcnn.literature
    },
    "data": { 
        # Dataset specific parameter
    },
    "training": {
        "fit": { 
            # keras fit arguments serialized
        },
        "compile": { 
            # Keras compile arguments serialized
        },
        "Kfold": {
            # kwargs unpacked in scikit-learn Kfold class.  
        }
    }
}
```

Furthermore, you could also have a dictionary of models as ``hyper={"GCN": {...}, ...}`` which each model has a config dictionary as above.

If a python file is used, also non-serialized hyper-parameter for fit and compile can be provided. 
However, note that the python file will be executed, and a serialization after model fit may fail depending on the arguments.