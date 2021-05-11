![GitHub release (latest by date)](https://img.shields.io/github/v/release/aimat-lab/gcnn_keras)
[![Documentation Status](https://readthedocs.org/projects/kgcnn/badge/?version=latest)](https://kgcnn.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/kgcnn.svg)](https://badge.fury.io/py/kgcnn)
![PyPI - Downloads](https://img.shields.io/pypi/dm/kgcnn)
[![kgcnn_unit_tests](https://github.com/aimat-lab/gcnn_keras/actions/workflows/unittests.yml/badge.svg)](https://github.com/aimat-lab/gcnn_keras/actions/workflows/unittests.yml)

# Keras Graph Convolutions

A set of layers for graph convolutions in TensorFlow Keras that use RaggedTensors.

# Table of Contents
* [General](#general)
* [Requirements](#requirements)
* [Installation](#installation)
* [Documentation](#documentation)
* [Implementation details](#implementation-details)
* [Literature](#literature)
* [Datasets](#datasets)
* [Examples](#examples)
* [Citing](#citing)
* [References](#references)
 

<a name="general"></a>
# General

The package in [kgcnn](kgcnn) contains several layer classes to build up graph convolution models. 
Some models are given as an example.
A [documentation](https://kgcnn.readthedocs.io/en/latest/index.html) is generated in [docs](docs).
This repo is still under construction.
Any comments, suggestions or help are very welcome!

<a name="requirements"></a>
# Requirements

For kgcnn, usually the latest version of tensorflow is required, but is listed as extra requirements in the `setup.py` for simplicity. 
Additional python packages are placed in the `setup.py` requirements and are installed automatically.
* tensorflow>=2.4.1

<a name="installation"></a>
# Installation

Clone repository https://github.com/aimat-lab/gcnn_keras and install with editable mode:

```bash
pip install -e ./gcnn_keras
```
or latest release via Python Package Index.
```bash
pip install kgcnn
```
<a name="documentation"></a>
# Documentation

Auto-documentation is generated at https://kgcnn.readthedocs.io/en/latest/index.html .

<a name="implementation-details"></a>
# Implementation details

### Representation
The most frequent usage for graph convolutions is either node or graph classifaction. As for their size, either a single large graph, e.g. citation network or small (batched) graphs like molecules have to be considered. 
Graphs can be represented by an index list of connections plus feature information. Typical quantities in tensor format to describe a graph are listed below.

* `nodes`: Nodelist of shape `(batch, N, F)` where `N` is the number of nodes and `F` is the node feature dimension.
* `edges`: Edgelist of shape `(batch, M, F)` where `M` is the number of edges and `F` is the edge feature dimension.
* `indices`: Connectionlist of shape `(batch, M, 2)` where `M` is the number of edges. The indices denote a connection of incoming i and outgoing j node as `(i,j)`.
* `state`: Graph state information of shape `(batch, F)` where `F` denotes the feature dimension.
 
A major issue for graphs is their flexible size and shape, when using mini-batches. Here, for a graph implementation in the spirit of keras, the batch dimension should be kept also in between layers. This is realized by using `RaggedTensor`.

Note: At the moment, most layers support also a disjoint representation of flatten values plus graph-id tensor `[values, partition]` in place of the `RaggedTensor` for comparison purposes. 
However, this will likely be removed in future versions, as `RaggedTensor` is intended be the only tensor representation passed to and within the model.


### Input

In order to input batched tensors of variable length with keras, either zero-padding plus masking or ragged and sparse tensors can be used. Morover for more flexibility, a dataloader from `tf.keras.utils.Sequence` is often used to input disjoint graph representations. Tools for converting numpy or scipy arrays are found in [utils](kgcnn/utils).

Here, for ragged tensors, the nodelist of shape `(batch, None, F)` and edgelist of shape `(batch, None, Fe)` have one ragged dimension `(None, )`.
The graph structure is represented by an index-list of shape `(batch, None, 2)` with index of incoming `i` and outgoing `j` node as `(i,j)`. 
The first index of incoming node `i` is usually expected to be sorted for faster pooling operations, but can also be unsorted (see layer arguments). Furthermore, the graph is directed, so an additional edge with `(j,i)` is required for undirected graphs. A ragged constant can be directly obtained from a list of numpy arrays: `tf.ragged.constant(indices,ragged_rank=1,inner_shape=(2,))` which yields shape `(batch, None, 2)`.

### Model

Models can be set up in a functional. Example message passing from fundamental operations:


```python
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.keras import Dense  # ragged support
from kgcnn.layers.pooling import PoolingLocalMessages

feature_dim = 10
n = ks.layers.Input(shape=(None,feature_dim),name='node_input',dtype ="float32",ragged=True)
ei = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)

n_in_out = GatherNodes()([n,ei])
node_messages = Dense(feature_dim)(n_in_out)
node_updates = PoolingLocalMessages()([n,node_messages,ei])
n_node_updates = ks.layers.Concatenate(axis=-1)([n,node_updates])
n_embedd = Dense(feature_dim)(n_node_updates)

message_passing = ks.models.Model(inputs=[n,ei], outputs=n_embedd)
```




<a name="literature"></a>
# Literature
A version of the following models are implemented in [literature](kgcnn/literature):
* **[GCN](kgcnn/literature/GCN.py)**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) by Kipf et al. (2016)
* **[INorp](kgcnn/literature/INorp.py)**: [Interaction Networks for Learning about Objects,Relations and Physics](https://arxiv.org/abs/1612.00222) by Battaglia et al. (2016)
* **[Megnet](kgcnn/literature/Megnet.py)**: [Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://doi.org/10.1021/acs.chemmater.9b01294) by Chen et al. (2019)
* **[NMPN](kgcnn/literature/NMPN.py)**: [Neural Message Passing for Quantum Chemistry](http://arxiv.org/abs/1704.01212) by Gilmer et al. (2017)
* **[Schnet](kgcnn/literature/Schnet.py)**: [SchNet – A deep learning architecture for molecules and materials ](https://aip.scitation.org/doi/10.1063/1.5019779) by Schütt et al. (2017)
* **[Unet](kgcnn/literature/Unet.py)**: [Graph U-Nets](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf) by H. Gao and S. Ji (2019)
* **[GNNExplainer](kgcnn/literature/GNNExplain.py)**: [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) by Ying et al. (2019)
* **[GraphSAGE](kgcnn/literature/GraphSAGE.py)**: [Inductive Representation Learning on Large Graphs](http://arxiv.org/abs/1706.02216) by Hamilton et al. (2017)
* **[GAT](kgcnn/literature/GAT.py)**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) by Veličković et al. (2018)

<a name="datasets"></a>
# Datasets

In [data](kgcnn/data) there are simple data handling tools that are used for examples.

<a name="examples"></a>
# Examples

A set of example traing can be found in [example](examples)

<a name="citing"></a>
# Citing

If you want to cite this repo, refer to our preprint:

```
@misc{reiser2021implementing,
      title={Implementing graph neural networks with TensorFlow-Keras}, 
      author={Patrick Reiser and Andre Eberhard and Pascal Friederich},
      year={2021},
      eprint={2103.04318},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<a name="references"></a>
# References

- https://www.tensorflow.org/api_docs/python/tf/RaggedTensor
