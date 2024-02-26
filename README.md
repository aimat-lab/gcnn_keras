![GitHub release (latest by date)](https://img.shields.io/github/v/release/aimat-lab/gcnn_keras)
[![Documentation Status](https://readthedocs.org/projects/kgcnn/badge/?version=latest)](https://kgcnn.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/kgcnn.svg)](https://badge.fury.io/py/kgcnn)
![PyPI - Downloads](https://img.shields.io/pypi/dm/kgcnn)
[![kgcnn_unit_tests](https://github.com/aimat-lab/gcnn_keras/actions/workflows/unittests.yml/badge.svg)](https://github.com/aimat-lab/gcnn_keras/actions/workflows/unittests.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.simpa.2021.100095%20-blue)](https://doi.org/10.1016/j.simpa.2021.100095)
![GitHub](https://img.shields.io/github/license/aimat-lab/gcnn_keras)
![GitHub issues](https://img.shields.io/github/issues/aimat-lab/gcnn_keras)
![Maintenance](https://img.shields.io/maintenance/yes/2024)

# Keras Graph Convolution Neural Networks
<p align="left">
  <img src="https://github.com/aimat-lab/gcnn_keras/blob/master/docs/source/_static/icon.svg" height="80"/>
</p>


[General](#general) | [Requirements](#requirements) | [Installation](#installation) | [Documentation](#documentation) | [Implementation details](#implementation-details)
 | [Literature](#literature) | [Data](#data)  | [Datasets](#datasets) | [Training](#training) | [Issues](#issues) | [Citing](#citing) | [References](#references)
 

<a name="general"></a>
# General

The package in [kgcnn](kgcnn) contains several layer classes to build up graph convolution models in 
Keras with Tensorflow, PyTorch or Jax as backend. 
Some models are given as an example in literature.
A [documentation](https://kgcnn.readthedocs.io/en/latest/index.html) is generated in [docs](docs).
Focus of [kgcnn](kgcnn) is (batched) graph learning for molecules [kgcnn.molecule](kgcnn/molecule) and materials [kgcnn.crystal](kgcnn/crystal).
If you want to get in contact, feel free to [discuss](https://github.com/aimat-lab/gcnn_keras/discussions). 

Note that kgcnn>=4.0.0 requires keras>=3.0.0. Previous versions of kgcnn were focused on ragged tensors of tensorflow, for which
hyperparameter for models should also transfer to kgcnn 4.0 by adding `input_tensor_type: "ragged"` and checking the order and *dtype* of inputs.

<a name="requirements"></a>
# Requirements

Standard python package requirements are installed automatically.
However, you must make sure to install the GPU/TPU acceleration for the backend of your choice.

<a name="installation"></a>
# Installation

Clone [repository](https://github.com/aimat-lab/gcnn_keras) or latest [release](https://github.com/aimat-lab/gcnn_keras/releases) and install with editable mode or latest release via [Python Package Index](https://pypi.org/project/kgcnn/).
```bash
pip install kgcnn
```
<a name="documentation"></a>
# Documentation

Auto-documentation is generated at https://kgcnn.readthedocs.io/en/latest/index.html .

<a name="implementation-details"></a>
# Implementation details

### Representation

A graph of `N` nodes and `M` edges is commonly represented by a list of node or edge attributes: `node_attr` or `edge_attr`, respectively. 
Plus a list of indices pairs `(i, j)` that represents a directed edge in the graph: `edge_index`. 
The feature dimension of the attributes is denoted by `F`. 
Alternatively, an adjacency matrix `A_ij` of shape `(N, N)` can be ascribed that has 'ones' entries
where there is an edge between nodes and 'zeros' elsewhere. Consequently, sum of `A_ij` will give `M` edges.

<a name="implementation-details-input"></a>
### Input

For learning on batches or single graphs, following tensor representation can be chosen:

###### Batched Graphs

* `node_attr`: Node attributes of shape `(batch, N, F)` and dtype *float*
* `edge_attr`: Edge attributes of shape `(batch, M, F)` and dtype *float*
* `edge_index`: Indices of shape `(batch, M, 2)` and dtype *int*
* `graph_attr`: Graph attributes of shape `(batch, F)` and dtype *float*

Graphs are stacked along the batch dimension `batch`. Note that for flexible sized graphs the tensor has to be padded up to a max `N`/`M` or ragged tensors are used,
with a ragged rank of one.

###### Disjoint Graphs

* `node_attr`: Node attributes of shape `([N], F)` and dtype *float*
* `edge_attr`: Edge attributes of shape `([M], F)` and dtype *float*
* `edge_index`: Indices of shape `(2, [M])` and dtype *int*
* `batch_ID`: Graph ID of shape `([N], )` and dtype *int*

Here, the lists essentially represent one graph but which consists of disjoint sub-graphs from the batch, 
which has been introduced by PytorchGeometric (PyG). 
For pooling, the graph assignment is stored in `batch_ID`. 
Note, that for Jax, we can not have dynamic shapes, so we use a padded disjoint representation assigning 
all padded nodes to a discarded graph with zero index.

### Model

The keras layers in [kgcnn.layers](kgcnn/layers) can be used with PyG compatible tensor representation. 
Or even by simply wrapping a PyG model with `TorchModuleWrapper`. Efficient model loading can be achieved 
in multiple ways (see [kgcnn.io](kgcnn/io)).
For most simple keras-like behaviour, the model can fed with batched padded or ragged tensor which are converted to/from
disjoint representation wrapping the PyG equivalent model.
Here an example of a minimal message passing GNN:

```python
import keras as ks
from kgcnn.layers.casting import CastBatchedIndicesToDisjoint
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.aggr import AggregateLocalEdges

# Example for padded input.
ns = ks.layers.Input(shape=(None, 64), dtype="float32", name="node_attributes")
e_idx = ks.layers.Input(shape=(None, 2), dtype="int64", name="edge_indices")
total_n = ks.layers.Input(shape=(), dtype="int64", name="total_nodes")  # Or mask
total_e = ks.layers.Input(shape=(), dtype="int64", name="total_edges")  # Or mask

n, idx, batch_id, _, _, _, _, _ = CastBatchedIndicesToDisjoint(uses_mask=False)([ns, e_idx, total_n, total_e])
n_in_out = GatherNodes()([n, idx])
node_messages = ks.layers.Dense(64, activation='relu')(n_in_out)
node_updates = AggregateLocalEdges()([n, node_messages, idx])
n_node_updates = ks.layers.Concatenate()([n, node_updates])
n_embedding = ks.layers.Dense(1)(n_node_updates)
g_embedding = PoolingNodes()([total_n, n_embedding, batch_id])

message_passing = ks.models.Model(inputs=[ns, e_idx, total_n, total_e], outputs=g_embedding)
```

The actual message passing model can further be structured by e.g. subclassing the message passing base layer:

```python
import keras as ks
from kgcnn.layers.message import MessagePassingBase

class MyMessageNN(MessagePassingBase):

    def __init__(self, units, **kwargs):
        super(MyMessageNN, self).__init__(**kwargs)
        self.dense = ks.layers.Dense(units)
        self.add = ks.layers.Add()

    def message_function(self, inputs, **kwargs):
        n_in, n_out, edges = inputs
        return self.dense(n_out, **kwargs)

    def update_nodes(self, inputs, **kwargs):
        nodes, nodes_update = inputs
        return self.add([nodes, nodes_update], **kwargs)
```

<a name="literature"></a>
# Literature
The following models, proposed in literature, have a module in [literature](kgcnn/literature). The module usually exposes a `make_model` function
to create a ``keras.models.Model``. The models can but must not be build completely from `kgcnn.layers` and can for example include
original implementations (with proper licencing).

* **[AttentiveFP](kgcnn/literature/AttentiveFP)**: [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959) by Xiong et al. (2019)
* **[CGCNN](kgcnn/literature/CGCNN)**: [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) by Xie et al. (2018)
* **[CMPNN](kgcnn/literature/CMPNN)**: [Communicative Representation Learning on Attributed Molecular Graphs](https://www.ijcai.org/proceedings/2020/0392.pdf) by Song et al. (2020)
* **[DGIN](kgcnn/literature/DGIN)**: [Improved Lipophilicity and Aqueous Solubility Prediction with Composite Graph Neural Networks ](https://pubmed.ncbi.nlm.nih.gov/34684766/) by Wieder et al. (2021)
* **[DimeNetPP](kgcnn/literature/DimeNetPP)**: [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115) by Klicpera et al. (2020)
* **[DMPNN](kgcnn/literature/DMPNN)**: [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) by Yang et al. (2019)
* **[EGNN](kgcnn/literature/EGNN)**: [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) by Satorras et al. (2021)
* **[GAT](kgcnn/literature/GAT)**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) by Veličković et al. (2018)

<details>
<summary> ... and many more <b>(click to expand)</b>.</summary>

* **[GATv2](kgcnn/literature/GATv2)**: [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) by Brody et al. (2021)
* **[GCN](kgcnn/literature/GCN)**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) by Kipf et al. (2016)
* **[GIN](kgcnn/literature/GIN)**: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) by Xu et al. (2019)
* **[GNNExplainer](kgcnn/literature/GNNExplain)**: [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) by Ying et al. (2019)
* **[GNNFilm](kgcnn/literature/GNNFilm)**: [GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation](https://arxiv.org/abs/1906.12192) by Marc Brockschmidt (2020)
* **[GraphSAGE](kgcnn/literature/GraphSAGE)**: [Inductive Representation Learning on Large Graphs](http://arxiv.org/abs/1706.02216) by Hamilton et al. (2017)
* **[HamNet](kgcnn/literature/HamNet)**: [HamNet: Conformation-Guided Molecular Representation with Hamiltonian Neural Networks](https://arxiv.org/abs/2105.03688) by Li et al. (2021)
* **[HDNNP2nd](kgcnn/literature/HDNNP2nd)**: [Atom-centered symmetry functions for constructing high-dimensional neural network potentials](https://aip.scitation.org/doi/abs/10.1063/1.3553717) by Jörg Behler (2011)
* **[INorp](kgcnn/literature/INorp)**: [Interaction Networks for Learning about Objects,Relations and Physics](https://arxiv.org/abs/1612.00222) by Battaglia et al. (2016)
* **[MAT](kgcnn/literature/MAT)**: [Molecule Attention Transformer](https://arxiv.org/abs/2002.08264) by Maziarka et al. (2020)
* **[MEGAN](kgcnn/literature/MEGAN)**: [MEGAN: Multi-explanation Graph Attention Network](https://link.springer.com/chapter/10.1007/978-3-031-44067-0_18) by Teufel et al. (2023)
* **[Megnet](kgcnn/literature/Megnet)**: [Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://doi.org/10.1021/acs.chemmater.9b01294) by Chen et al. (2019)
* **[MoGAT](kgcnn/literature/MoGAT)**: [Multi-order graph attention network for water solubility prediction and interpretation](https://www.nature.com/articles/s41598-022-25701-5) by Lee et al. (2023)
* **[MXMNet](kgcnn/literature/MXMNet)**: [Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures](https://arxiv.org/abs/2011.07457) by Zhang et al. (2020)
* **[NMPN](kgcnn/literature/NMPN)**: [Neural Message Passing for Quantum Chemistry](http://arxiv.org/abs/1704.01212) by Gilmer et al. (2017)
* **[PAiNN](kgcnn/literature/PAiNN)**: [Equivariant message passing for the prediction of tensorial properties and molecular spectra](https://arxiv.org/pdf/2102.03150.pdf) by Schütt et al. (2020)
* **[RGCN](kgcnn/literature/RGCN)**: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) by Schlichtkrull et al. (2017)
* **[rGIN](kgcnn/literature/rGIN)** [Random Features Strengthen Graph Neural Networks](https://arxiv.org/abs/2002.03155) by Sato et al. (2020)
* **[Schnet](kgcnn/literature/Schnet)**: [SchNet – A deep learning architecture for molecules and materials ](https://aip.scitation.org/doi/10.1063/1.5019779) by Schütt et al. (2017)

</details>


<a name="data"></a>
# Data

Data handling classes are given in `kgcnn.data` which stores graphs as `List[Dict]` .

#### Graph dictionary

Graphs are represented by a dictionary `GraphDict` of (numpy) arrays which behaves like a python `dict`.
There are graph pre- and postprocessors in ``kgcnn.graph`` which take specific properties by name and apply a
processing function or transformation. 

> [!IMPORTANT]  
> They can do any operation but note that `GraphDict` does not impose an actual graph structure!
> For example to sort edge indices make sure that all attributes are sorted accordingly. 


```python
from kgcnn.graph import GraphDict
# Single graph.
graph = GraphDict({"edge_indices": [[1, 0], [0, 1]], "node_label": [[0], [1]]})
graph.set("graph_labels", [0])  # use set(), get() to assign (tensor) properties.
graph.set("edge_attributes", [[1.0], [2.0]])
graph.to_networkx()
# Modify with e.g. preprocessor.
from kgcnn.graph.preprocessor import SortEdgeIndices
SortEdgeIndices(edge_indices="edge_indices", edge_attributes="^edge_(?!indices$).*", in_place=True)(graph)
```

#### List of graph dictionaries

A `MemoryGraphList` should behave identical to a python list but contain only `GraphDict` items.

```python
from kgcnn.data import MemoryGraphList
# List of graph dicts.
graph_list = MemoryGraphList([{"edge_indices": [[0, 1], [1, 0]]}, {"edge_indices": [[0, 0]]}, {}])
graph_list.clean(["edge_indices"])  # Remove graphs without property
graph_list.get("edge_indices")  # opposite is set()
# Easily cast to tensor; makes copy.
tensor = graph_list.tensor([{"name": "edge_indices"}])  # config of keras `Input` layer
# Or directly modify list.
for i, x in enumerate(graph_list):
    x.set("graph_number", [i])
print(len(graph_list), graph_list[:2])  # Also supports indexing lists.
```


<a name="datasets"></a>
# Datasets

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
from kgcnn.data import MemoryGraphDataset
dataset = MemoryGraphDataset(data_directory="ExampleDir/", 
                             dataset_name="Example",
                             file_name=None, file_directory=None)
dataset.save()  # opposite is load(). 
```

The subclasses `QMDataset`, `ForceDataset`, `MoleculeNetDataset`, `CrystalDataset` and `GraphTUDataset` further have functions required for the specific dataset type to convert and process files such as '.txt', '.sdf', '.xyz' etc. 
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

In [data.datasets](kgcnn/data/datasets) there are graph learning benchmark datasets as subclasses which are being *downloaded* from e.g. popular graph archives like [TUDatasets](https://chrsmrrs.github.io/datasets/), [MatBench](https://matbench.materialsproject.org/) or [MoleculeNet](https://moleculenet.org/). 
The subclasses `GraphTUDataset2020`, `MatBenchDataset2020` and `MoleculeNetDataset2018` download and read the available datasets by name.
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
* Jagged or nested Tensors loading into models for PyTorch backend is not working.
* BatchNormalization layer dos not support padding yet.
* Keras AUC metric does not seem to work for torch cuda.

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
