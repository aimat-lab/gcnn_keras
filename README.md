# Keras GCN

A set of Layer classes for tensorflow keras to handle graph convolutions.

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
Any comments or suggestions are very welcome.

<a name="installation"></a>
# Installation

Clone repository and install with editable mode:

```bash
pip install -e ./gcnn_keras
```

<a name="implementation-details"></a>
# Implementation details

The major issue for graphs is their flexible size and shape. To handle flexible input tensors with keras,
either zero-padding plus masking or ragged/sparse tensors can be used.

* Ragged Tensor
Ragged Tensors already come with a flexible batch-dimension and are most suited for graphs. However Keras support for ragged tensor is limited for the moment.
The ragged Tensor is simply a flatten tensor plus row_length index.

* Sparse Tensor

<a name="datasets"></a>
# Datasets

In [data](kgcnn/data) there are simple file handling classes

<a name="examples"></a>
# Examples

A set of example traing can be found in [example](examples)