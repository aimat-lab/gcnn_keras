# Training

Example training to test the performance of model implementations of ``kgcnn`` per datasets. The training scripts are called via:

```bash
python3 train_qm9.py
python3 train_qm9.py --model Schnet
python3 train_qm9.py --model Schnet --hpyer my_config.json
```

Where `my_config.json` stores custom hyper-parameters and must be in the same folder or a path to a `.json` file. 
Alternatively, also a `.yaml`, `.yaml` or `.py` file can be loaded in place of the `.json` file. 
The python file must define a ```hyper``` attribute as described below.
If no config file is provided a default for each model is used. 
However, note that not all models can be trained on all datasets and that not all models have proper default hyper parameters.
You can check previous runs in the result folders named after each dataset and their hyper parameters and output files.

There is a [``make_config_training.ipynb``](/notebooks/make_config_training.ipynb) jupyter [notebook](/notebooks) to demonstrate how the `.json` config file can be set up and provide further information
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