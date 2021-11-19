# Implementation details

The layers for `kgcnn` should accept ragged tensor input and are sorted as following: 

* The most general layers that accept all kind of tensor input and kept maintained beyond different models are located in `kgcnn.layers`. These are:
    * `kgcnn.layers.gather` Layers around tf.gather
    * `kgcnn.layers.pooling` General layers for standard aggregation and pooling.
* Model specific pooling and convolutional layers are sorted into:
    * `kgcnn.layers.pool`
    * `kgcnn.layers.conv`

Thereby it should be possible to contribute to `kgcnn` by supplying new layers in `conv` and `pool` and a 
corresponding model in `kgcnn.literature`. Naming should be made following the existing convention.