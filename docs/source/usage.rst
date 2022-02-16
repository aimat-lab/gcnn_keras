.. _usage:
   :maxdepth: 3

Usage
=====

Notebooks
---------

Examples and tutorials for using modules and classes of ``kgcnn`` are furthermore added as jupyter `notebooks <https://github.com/aimat-lab/gcnn_keras/tree/master/notebooks>`_.

Training
--------

Example training to test the performance of model implementations of ``kgcnn`` per datasets. The training scripts are called via::


    python3 train_citation.py --dataset CoraLuDataset --model GAT --hyper hyper/hyper_esol.py
    python3 train_moleculenet.py --dataset ESOLDataset --model GIN --hyper hyper/hyper_esol.py
    python3 train_tudataset.py --dataset PROTEINSDataset --model GIN --hyper hyper/hyper_proteins.py


Where ``hyper_esol.py`` stores hyperparameter and must be in the same folder or a path to a `.py`.
Alternatively, also a `.yaml`, `.yaml` or `.json` file can be loaded in place of the `.py` file.
The python file must define a ``hyper`` attribute as described below.
However, note that not all models can be trained on all datasets and that not all models have proper default hyperparameter here.
You can check previous runs in the result folders named after each dataset and their hyperparameter and output files.

There is a `make_config_training.ipynb <https://github.com/aimat-lab/gcnn_keras/blob/master/notebooks/tutorial_config_training.ipynb>`_ jupyter `notebook <https://github.com/aimat-lab/gcnn_keras/tree/master/notebooks>`_ to demonstrate how
the ``.py`` config file can be set up and provide further information
on how it is structured. In short the config file contains a python dictionary of the form::


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


Furthermore, you could also have a dictionary of models as ``hyper={"GCN": {...}, ...}`` which each model has a config dictionary as above.

If a python file is used, also non-serialized hyperparameter for fit and compile can be provided.
However, note that the python file will be executed, and a serialization after model fit may fail depending on the arguments.

Literature
----------

A version of the following models is implemented in [literature](kgcnn/literature):

* `GCN <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/GCN.py>`_: `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/abs/1609.02907>`_ by Kipf et al. (2016)
* `INorp <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/INorp.py>`_: `Interaction Networks for Learning about Objects,Relations and Physics <https://arxiv.org/abs/1612.00222>`_ by Battaglia et al. (2016)
* `Megnet <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/Megnet.py>`_: `Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals <https://doi.org/10.1021/acs.chemmater.9b01294>`_ by Chen et al. (2019)
* `NMPN <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/NMPN.py>`_: `Neural Message Passing for Quantum Chemistry <http://arxiv.org/abs/1704.01212>`_ by Gilmer et al. (2017)
* `Schnet <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/Schnet.py>`_: `SchNet – A deep learning architecture for molecules and materials <https://aip.scitation.org/doi/10.1063/1.5019779>`_ by Schütt et al. (2017)
* `Unet <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/Unet.py>`_: `Graph U-Nets <http://proceedings.mlr.press/v97/gao19a/gao19a.pdf>`_ by H. Gao and S. Ji (2019)
* `GNNExplainer <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/GNNExplain.py>`_: `GNNExplainer: Generating Explanations for Graph Neural Networks <https://arxiv.org/abs/1903.03894>`_ by Ying et al. (2019)
* `GraphSAGE <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/GraphSAGE.py>`_: `Inductive Representation Learning on Large Graphs <http://arxiv.org/abs/1706.02216>`_ by Hamilton et al. (2017)
* `GAT <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/GAT.py>`_: `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`_ by Veličković et al. (2018)
* `GATv2 <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/GATv2.py>`_: `How Attentive are Graph Attention Networks? <https://arxiv.org/abs/2105.14491>`_ by Brody et al. (2021)
* `DimeNetPP <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/DimeNetPP.py>`_: `Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules <https://arxiv.org/abs/2011.14115>`_ by Klicpera et al. (2020)
* `AttentiveFP <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/AttentiveFP.py>`_: `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ by Xiong et al. (2019)
* `GIN <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature//IN.py>`_: `How Powerful are Graph Neural Networks? <https://arxiv.org/abs/1810.00826>`_ by Xu et al. (2019)
* `PAiNN <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/PAiNN.py>`_: `Equivariant message passing for the prediction of tensorial properties and molecular spectra] <https://arxiv.org/pdf/2102.03150.pdf>`_ by Schütt et al. (2020)
* `DMPNN <https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/DMPNN.py>`_: `Analyzing Learned Molecular Representations for Property Prediction <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237>`_ by Yang et al. (2019)

