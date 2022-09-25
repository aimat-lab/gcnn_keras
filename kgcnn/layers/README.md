# Implementation details

The layers for `kgcnn` should accept ragged tensor input and are sorted as following: 

* The most general layers that kept maintained beyond different models with proper documentation are located in `kgcnn.layers`. These are:
    * `kgcnn.layers.casting` Layers for casting tensor formats.
    * `kgcnn.layers.gather` Layers around tf.gather.
    * `kgcnn.layers.geom` Geometry operations.
    * `kgcnn.layers.mlp` Multi-layer perceptron for graphs.
    * `kgcnn.layers.norm` Normalization layers for graph tensors. 
    * `kgcnn.layers.modules` Keras layers and modules to support ragged tensor input.
    * `kgcnn.layers.pooling` General layers for standard aggregation and pooling.
    * `kgcnn.layers.message` Message passing base layer.


* Model specific pooling and convolutional layers (they should make use of existing modules in `kgcnn.layers`) are sorted into:
    * `kgcnn.layers.pool`
    * `kgcnn.layers.conv`


Thereby it should be possible to contribute to `kgcnn` by supplying new layers in `kgcnn.layers.pool` and `kgcnn.layers.conv` and a 
corresponding model in `kgcnn.literature`. Naming and implementation should be made following the existing convention, if possible,
which is model name + '_conv' or '_pool'.