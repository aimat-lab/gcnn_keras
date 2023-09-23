![GitHub release (latest by date)](https://img.shields.io/github/v/release/aimat-lab/gcnn_keras)
[![Documentation Status](https://readthedocs.org/projects/kgcnn/badge/?version=latest)](https://kgcnn.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/kgcnn.svg)](https://badge.fury.io/py/kgcnn)
![PyPI - Downloads](https://img.shields.io/pypi/dm/kgcnn)
[![kgcnn_unit_tests](https://github.com/aimat-lab/gcnn_keras/actions/workflows/unittests.yml/badge.svg)](https://github.com/aimat-lab/gcnn_keras/actions/workflows/unittests.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.simpa.2021.100095%20-blue)](https://doi.org/10.1016/j.simpa.2021.100095)
![GitHub](https://img.shields.io/github/license/aimat-lab/gcnn_keras)
![GitHub issues](https://img.shields.io/github/issues/aimat-lab/gcnn_keras)
![Maintenance](https://img.shields.io/maintenance/yes/2023)

# Keras Graph Convolution Neural Networks
<p align="left">
  <img src="https://github.com/aimat-lab/gcnn_keras/blob/master/docs/source/_static/icon.svg" height="80"/>
</p>

A set of layers for graph convolutions in Keras.

> [!IMPORTANT]  
> The versions of kgcnn<=3.1.0 were focused on ragged tensors of tensorflow.
> The current main branch is developped for Keras 3.0 . Please use last release of 3.1.0 for previous version of kgcnn.

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

<a name="requirements"></a>
# Requirements

Standard python package requirements are placed in the `setup.py` and are installed automatically.
However, you must make sure to install the GPU/TPU acceleration for the backend of your choice.

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

TODO

<a name="implementation-details-input"></a>
### Input

TODO


### Model

Models can be set up in a functional way. Example message passing from fundamental operations:

TODO

<a name="literature"></a>
# Literature
The following models, proposed in literature, have a module in [literature](kgcnn/literature). The module usually exposes a `make_model` function
to create a ``tf.keras.models.Model``, which features ragged tensor in- or output. The models can but must not be build completely from `kgcnn.layers` and can for example include
original implementations (with proper licencing).

* **[GCN](kgcnn/literature/GCN)**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) by Kipf et al. (2016)
* **[Schnet](kgcnn/literature/Schnet)**: [SchNet – A deep learning architecture for molecules and materials ](https://aip.scitation.org/doi/10.1063/1.5019779) by Schütt et al. (2017)
* **[GAT](kgcnn/literature/GAT)**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) by Veličković et al. (2018)
* **[GraphSAGE](kgcnn/literature/GraphSAGE)**: [Inductive Representation Learning on Large Graphs](http://arxiv.org/abs/1706.02216) by Hamilton et al. (2017)
* **[GIN](kgcnn/literature/GIN)**: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) by Xu et al. (2019)

<details>
<summary> ... and many more <b>(click to expand)</b>.</summary>

* **[GATv2](kgcnn/literature/GATv2)**: [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) by Brody et al. (2021)

</details>


<a name="data"></a>
# Data

Data handling classes are given in `kgcnn.data` which stores graphs as `List[Dict]` .

#### Graph dictionary

Graphs are represented by a dictionary `GraphDict` of (numpy) arrays which behaves like a python `dict`.
There are graph pre- and postprocessors in ``kgcnn.graph`` which take specific properties by name and apply a
processing function or transformation.

```python
from kgcnn.data.base import GraphDict
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
from kgcnn.data.base import MemoryGraphList
# List of graph dicts.
graph_list = MemoryGraphList([{"edge_indices": [[0, 1], [1, 0]]}, {"edge_indices": [[0, 0]]}, {}])
graph_list.clean(["edge_indices"])  # Remove graphs without property
graph_list.get("edge_indices")  # opposite is set()
# Easily cast to (ragged) tf-tensor; makes copy.
tensor = graph_list.tensor([{"name": "edge_indices", "ragged": True}])  # config of keras `Input` layer
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
from kgcnn.data.base import MemoryGraphDataset
dataset = MemoryGraphDataset(data_directory="ExampleDir/", 
                             dataset_name="Example",
                             file_name=None, file_directory=None)
dataset.save()  # opposite is load(). 
```

The subclasses `QMDataset`, `ForceDataset`, `MoleculeNetDataset`, `CrystalDataset`, `VisualGraphDataset` and `GraphTUDataset` further have functions required for the specific dataset type to convert and process files such as '.txt', '.sdf', '.xyz' etc. 
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
* TODO  

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
