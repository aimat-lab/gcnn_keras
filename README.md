# Keras Graph Convolutions

A set of layers for graph convolutions in tensorflow keras.

# Table of Contents
* [General](#general)
* [Implementation details](#implementation-details)
* [Installation](#installation)
* [Datasets](#datasets)
* [Examples](#examples)
* [Tests](#tests)
 

<a name="general"></a>
# General

The package in [kgcnn](kgcnn) contains several layer classes to build up graph convolution models. 
Some models are given as an example.
A documentation is generated in [docs](docs).
This repo is still under construction.
!! Any comments, suggestions or help are very welcome !! 

<a name="installation"></a>
# Installation

Clone repository and install with editable mode:

```bash
pip install -e ./gcnn_keras
```

<a name="implementation-details"></a>
# Implementation details

The major issue for graphs is their flexible size and shape, when using mini-batches. To handle flexible input tensors with keras,
either zero-padding plus masking or ragged/sparse tensors can be used. 
Depending on the task those representations can also be combined by casting from one to the other.
For more flexibility and a flatten batch-dimension, a dataloader from tf.keras.utils.Sequence is typically used. 

* Ragged Tensor:
Here the nodelist of shape `(batch,None,nodefeatures)` and edgelist of shape `(batch,None,edgefeatures)` are given by ragged tensors with ragged dimension (None,).
The graph structure is represented by an indexlist of shape `(batch,None,2)` with index of incoming i and outgoing j node as `(i,j)`. 
The first index of incoming node i is expected to be sorted for faster pooling opertions. Furthermore the graph is directed, so an additional edge with `(j,i)` is required for undirected graphs.
In principle also the adjacency matrix can be represented as ragged tensor of shape `(batch,None,None)` but will be dense within each graph.

* Padded Tensor:
The node- and edgelists are given by a full-rank tensor of shape (batch,Nmax,features) with Nmax being the maximum number of edges or nodes in the dataset, 
by padding all unused entries with zero and marking them in an additional mask tensor of shape (batch,Nmax). 
This is only practical for highly connected graphs of similar shapes. 
Applying the mask is done by simply multiplying with the mask tensor. For pooling layers tf.boolean_mask() may be slower but can be favourable.
Besides the adjacencymatrix also the index list can be arranged in a matrix form with a max number of edges for faster node pooling, e.g. (batch,N,M) with number of nodes N and edges per Node M.


* Sparse Tensor:
...


<a name="datasets"></a>
# Datasets

In [data](kgcnn/data) there are simple file handling classes

<a name="examples"></a>
# Examples

A set of example traing can be found in [example](examples)