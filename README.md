![GitHub release (latest by date)](https://img.shields.io/github/v/release/aimat-lab/gcnn_keras)
[![Documentation Status](https://readthedocs.org/projects/kgcnn/badge/?version=latest)](https://kgcnn.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/kgcnn.svg)](https://badge.fury.io/py/kgcnn)
![PyPI - Downloads](https://img.shields.io/pypi/dm/kgcnn)
[![kgcnn_unit_tests](https://github.com/aimat-lab/gcnn_keras/actions/workflows/unittests.yml/badge.svg)](https://github.com/aimat-lab/gcnn_keras/actions/workflows/unittests.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.simpa.2021.100095%20-blue)](https://doi.org/10.1016/j.simpa.2021.100095)
![GitHub](https://img.shields.io/github/license/aimat-lab/gcnn_keras)
![GitHub issues](https://img.shields.io/github/issues/aimat-lab/gcnn_keras)
![Maintenance](https://img.shields.io/maintenance/yes/2022)

# Keras Graph Convolution Neural Networks
<p align="left">
  <img src="https://github.com/aimat-lab/gcnn_keras/blob/master/docs/source/_static/icon.svg" height="80"/>
</p>

A set of layers for graph convolutions in TensorFlow Keras that use RaggedTensors.

[General](#general) | [Requirements](#requirements) | [Installation](#installation) | [Documentation](#documentation) | [Implementation details](#implementation-details)
 | [Literature](#literature) | [Datasets](#datasets) | [Training](#training) | [Issues](#issues) | [Citing](#citing) | [References](#references)
 

<a name="general"></a>
# General

The package in [kgcnn](kgcnn) contains several layer classes to build up graph convolution models. 
Some models are given as an example.
A [documentation](https://kgcnn.readthedocs.io/en/latest/index.html) is generated in [docs](docs).
Focus of [kgcnn](kgcnn) is (batched) graph learning for molecules [kgcnn.mol](kgcnn/mol) and materials [kgcnn.crystal](kgcnn/crystal).
If you want to get in contact, feel free to [discuss](https://github.com/aimat-lab/gcnn_keras/discussions). 

<a name="requirements"></a>
# Requirements

For kgcnn, usually the latest version of tensorflow is required, but is listed as extra requirements in the `setup.py` for simplicity. 
Standard python package requirements are placed in the `setup.py` and are installed automatically. 
Packages which must be installed manually for full functionality:

* [tensorflow](https://www.tensorflow.org/install) >=2.4.1
* [rdkit](https://www.rdkit.org/docs/Install.html) >=2020.03.4
* [openbabel](http://openbabel.org/wiki/Main_Page) >=3.0.1
* [pymatgen](https://pymatgen.org/installation.html) >=2022.0.12

<a name="installation"></a>
# Installation

Clone [repository](https://github.com/aimat-lab/gcnn_keras) or latest [release](https://github.com/aimat-lab/gcnn_keras/releases) and install with editable mode:

```bash
pip install -e ./gcnn_keras
```
or latest release via [Python Package Index](https://pypi.org/project/kgcnn/).
```bash
pip install kgcnn
```
<a name="documentation"></a>
# Documentation

Auto-documentation is generated at https://kgcnn.readthedocs.io/en/latest/index.html .

<a name="implementation-details"></a>
# Implementation details

### Representation
The most frequent usage for graph convolutions is either node or graph classification. As for their size, either a single large graph, e.g. citation network or small (batched) graphs like molecules have to be considered. 
Graphs can be represented by an index list of connections plus feature information. Typical quantities in tensor format to describe a graph are listed below.

* `nodes`: Node-list of shape `(batch, [N], F)` where `N` is the number of nodes and `F` is the node feature dimension.
* `edges`: Edge-list of shape `(batch, [M], F)` where `M` is the number of edges and `F` is the edge feature dimension.
* `indices`: Connection-list of shape `(batch, [M], 2)` where `M` is the number of edges. The indices denote a connection of incoming or receiving node `i` and outgoing or sending node `j` as `(i, j)`.
* `state`: Graph state information of shape `(batch, F)` where `F` denotes the feature dimension.
 
A major issue for graphs is their flexible size and shape, when using mini-batches. Here, for a graph implementation in the spirit of keras, the batch dimension should be kept also in between layers. This is realized by using `RaggedTensor`s.

<a name="implementation-details-input"></a>
### Input

Graph tensors for edge-indices or attributes for multiple graphs is passed to the model in form of ragged tensors 
of shape `(batch, None, Dim)` where `Dim` denotes a fixed feature or index dimension.
Such a ragged tensor has `ragged_rank=1` with one ragged dimension indicated by `None` and is build from a value plus partition tensor.
For example, the graph structure is represented by an index-list of shape `(batch, None, 2)` with index of incoming or receiving node `i` and outgoing or sending node `j` as `(i, j)`.
Note, an additional edge with `(j, i)` is required for undirected graphs. 
A ragged constant can be easily created and passed to a model:

```python
import tensorflow as tf
import numpy as np
idx = [[[0, 1], [1, 0]], [[0, 1], [1, 2], [2, 0]], [[0, 0]]]  # batch_size=3
# Get ragged tensor of shape (3, None, 2)
print(tf.ragged.constant(idx, ragged_rank=1, inner_shape=(2, )).shape)
print(tf.RaggedTensor.from_row_lengths(np.concatenate(idx), [len(i) for i in idx]).shape) 
```


### Model

Models can be set up in a functional way. Example message passing from fundamental operations:

```python
import tensorflow as tf
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate  # ragged support
from kgcnn.layers.pooling import PoolingLocalMessages, PoolingNodes
ks = tf.keras

n = ks.layers.Input(shape=(None, 3), name='node_input', dtype="float32", ragged=True)
ei = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

n_in_out = GatherNodes()([n, ei])
node_messages = DenseEmbedding(10, activation='relu')(n_in_out)
node_updates = PoolingLocalMessages()([n, node_messages, ei])
n_node_updates = LazyConcatenate(axis=-1)([n, node_updates])
n_embedding = DenseEmbedding(1)(n_node_updates)
g_embedding = PoolingNodes()(n_embedding)

message_passing = ks.models.Model(inputs=[n, ei], outputs=g_embedding)
```

or via sub-classing of the message passing base layer. Where only `message_function` and `update_nodes` must be implemented:

```python

from kgcnn.layers.message import MessagePassingBase
from kgcnn.layers.modules import DenseEmbedding, LazyAdd


class MyMessageNN(MessagePassingBase):

    def __init__(self, units, **kwargs):
        super(MyMessageNN, self).__init__(**kwargs)
        self.dense = DenseEmbedding(units)
        self.add = LazyAdd(axis=-1)

    def message_function(self, inputs, **kwargs):
        n_in, n_out, edges = inputs
        return self.dense(n_out)

    def update_nodes(self, inputs, **kwargs):
        nodes, nodes_update = inputs
        return self.add([nodes, nodes_update])
```

<a name="literature"></a>
# Literature
A version of the following models and variants thereof are implemented in [literature](kgcnn/literature):
* **[GCN](kgcnn/literature/GCN.py)**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) by Kipf et al. (2016)
* **[Schnet](kgcnn/literature/Schnet.py)**: [SchNet – A deep learning architecture for molecules and materials ](https://aip.scitation.org/doi/10.1063/1.5019779) by Schütt et al. (2017)
* **[GAT](kgcnn/literature/GAT.py)**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) by Veličković et al. (2018)
* **[GraphSAGE](kgcnn/literature/GraphSAGE.py)**: [Inductive Representation Learning on Large Graphs](http://arxiv.org/abs/1706.02216) by Hamilton et al. (2017)
* **[DimeNetPP](kgcnn/literature/DimeNetPP.py)**: [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115) by Klicpera et al. (2020)
* **[GNNExplainer](kgcnn/literature/GNNExplain.py)**: [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) by Ying et al. (2019)
* **[AttentiveFP](kgcnn/literature/AttentiveFP.py)**: [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959) by Xiong et al. (2019)

<details>
<summary> ... and many more <b>(click to expand)</b>.</summary>

* **[INorp](kgcnn/literature/INorp.py)**: [Interaction Networks for Learning about Objects,Relations and Physics](https://arxiv.org/abs/1612.00222) by Battaglia et al. (2016)
* **[Megnet](kgcnn/literature/Megnet.py)**: [Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://doi.org/10.1021/acs.chemmater.9b01294) by Chen et al. (2019)
* **[NMPN](kgcnn/literature/NMPN.py)**: [Neural Message Passing for Quantum Chemistry](http://arxiv.org/abs/1704.01212) by Gilmer et al. (2017)
* **[Unet](kgcnn/literature/Unet.py)**: [Graph U-Nets](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf) by H. Gao and S. Ji (2019)
* **[GATv2](kgcnn/literature/GATv2.py)**: [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) by Brody et al. (2021)
* **[GIN](kgcnn/literature/GIN.py)**: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) by Xu et al. (2019)
* **[PAiNN](kgcnn/literature/PAiNN.py)**: [Equivariant message passing for the prediction of tensorial properties and molecular spectra](https://arxiv.org/pdf/2102.03150.pdf) by Schütt et al. (2020)
* **[DMPNN](kgcnn/literature/DMPNN.py)**: [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) by Yang et al. (2019)
* **[HamNet](kgcnn/literature/HamNet.py)**: [HamNet: Conformation-Guided Molecular Representation with Hamiltonian Neural Networks](https://arxiv.org/abs/2105.03688) by Li et al. (2021)
* **[CGCNN](kgcnn/literature/CGCNN.py)**: [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) by Xie et al. (2018)
* **[CMPNN](kgcnn/literature/CMPNN.py)**: [Communicative Representation Learning on Attributed Molecular Graphs](https://www.ijcai.org/proceedings/2020/0392.pdf) by Song et al. (2020)
* **[EGNN](kgcnn/literature/EGNN.py)**: [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) by Satorras et al. (2021)
* **[MAT](kgcnn/literature/MAT.py)**: [Molecule Attention Transformer](https://arxiv.org/abs/2002.08264) by Maziarka et al. (2020)
</details>

<a name="datasets"></a>
# Datasets

How to construct ragged tensors is shown [above](#implementation-details-input). 
Moreover, some data handling classes are given in `kgcnn.data`.
Graphs are represented by a dictionary of (numpy) tensors `GraphDict` and are stored in a list `MemoryGraphList`. 
Both must fit into memory and are supposed to be handled just like a python dict or list, respectively.

```python
from kgcnn.data.base import GraphDict, MemoryGraphList
# Single graph.
graph = GraphDict({"edge_indices": [[0, 1], [1, 0]]})
print(graph)
# List of graph dicts.
graph_list = MemoryGraphList([graph, {"edge_indices": [[0, 0]]}, {}])
graph_list.clean(["edge_indices"])  # Remove graphs without property
graph_list.get("edge_indices")  # opposite is set()
graph_list.tensor([{"name": "edge_indices", "ragged": True}]) # config of layers.Input; makes copy.
```

The `MemoryGraphDataset` inherits from `MemoryGraphList` but must be initialized with file information on disk that points to a `data_directory` for the dataset.
The `data_directory` can have a subdirectory for files and/or single file such as a CSV file: 

```bash
├── data_directory
    ├── file_directory
    │   ├── *.*
    │   └── ... 
    ├── file_name
    └── dataset_name.kgcnn.pickle
```
A base dataset class is created with path and name information:

```python
from kgcnn.data.base import MemoryGraphDataset
dataset = MemoryGraphDataset(data_directory="ExampleDir/", 
                             dataset_name="Example",
                             file_name=None, file_directory=None)
```

The subclasses `QMDataset`, `MoleculeNetDataset` and `GraphTUDataset` further have functions required for the specific dataset type to convert and process files such as '.txt', '.sdf', '.xyz' etc. 
Most subclasses implement `prepare_data()` and `read_in_memory()` with dataset dependent arguments.
An example for `MoleculeNetDataset` is shown below. 
For more details find tutorials in [notebooks](notebooks).

```python
from kgcnn.data.moleculenet import MoleculeNetDataset
# File directory and files must exist. 
# Here 'ExampleDir' and 'ExampleDir/data.csv' with columns "smiles" and "label".
dataset = MoleculeNetDataset(dataset_name="Example",
                             data_directory="ExampleDir/",
                             file_name="data.csv")
dataset.prepare_data(overwrite=True, smiles_column_name="smiles", add_hydrogen=True,
                     make_conformers=True, optimize_conformer=True, num_workers=None)
dataset.read_in_memory(label_column_name="label", add_hydrogen=False, 
                       has_conformers=True)
```

In [data.datasets](kgcnn/data/datasets) there are graph learning benchmark datasets as subclasses which are being *downloaded* from e.g. popular graph archives like [TUDatasets](https://chrsmrrs.github.io/datasets/) or [MoleculeNet](https://moleculenet.org/). 
The subclasses `GraphTUDataset2020` and `MoleculeNetDataset2018` download and read the available datasets by name.
There are also specific dataset subclasses for each dataset to handle additional processing or downloading from individual sources:

```python
from kgcnn.data.datasets.MUTAGDataset import MUTAGDataset
dataset = MUTAGDataset()  # inherits from GraphTUDataset2020
```

Downloaded datasets are stored in `~/.kgcnn/datasets` on your computer. Please remove them manually, if no longer required.

<a name="training"></a>
# Training

A set of example training can be found in [training](training). Training scripts are configurable with a hyperparameter config file and command line arguments regarding model and dataset.

You can find a [table](training/results/README.md) of common benchmark datasets in [results](training/results).

# Issues

Some known issues to be aware of, if using and making new models or layers with `kgcnn`.
* RaggedTensor can not yet be used as a keras model output [(issue)](https://github.com/tensorflow/tensorflow/issues/42320), which has been mostly resolved in TF 2.8.
* Using `RaggedTensor`'s for arbitrary ragged rank apart from `kgcnn.layers.modules` can cause significant performance decrease. This is due to shape check during add, multiply or concatenate (we think). 
  We therefore use lazy add and concat in the `kgcnn.layers.modules` layers or directly operate on the value tensor for possible rank. 
* With tensorflow version <=2.5 there is a problem with numpy version >=1.20 also affect `kgcnn` [(issue)](https://github.com/tensorflow/tensorflow/issues/47691) 

<a name="citing"></a>
# Citing

If you want to cite this repo, please refer to our [paper](https://doi.org/10.1016/j.simpa.2021.100095):

```
@article{REISER2021100095,
title = {Graph neural networks in TensorFlow-Keras with RaggedTensor representation (kgcnn)},
journal = {Software Impacts},
pages = {100095},
year = {2021},
issn = {2665-9638},
doi = {https://doi.org/10.1016/j.simpa.2021.100095},
url = {https://www.sciencedirect.com/science/article/pii/S266596382100035X},
author = {Patrick Reiser and Andre Eberhard and Pascal Friederich}
}
```

<a name="references"></a>
# References

- https://www.tensorflow.org/api_docs/python/tf/RaggedTensor
