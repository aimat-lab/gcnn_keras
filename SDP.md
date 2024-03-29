Software Development Plan (SDP)

* [ ] Add GemNet from original implementation.
* [x] Make flexible charge option in ``kgcnn.molecule.convert`` .
* [ ] Test and improve code for ``kgcnn.crystal`` . 
* [ ] Make pretty loop update in `kgcnn.datasets` .
* [x] Rework and simplify training scripts.
* [x] Add graph preprocessor from standard dictionary scheme also for ``crystal`` and `molecule` .
* [x] Rework and clean base layers.
* [ ] Add a properly designed transformer layer in ``kgcnn.layers`` .
* [ ] Add an element-wise loader for ``Graphlist`` apart from tensor files. Must change dataformat for standard save.
* [x] Make a ``tf_dataset()`` function to return a generator dataset from `Graphlist` .
* [ ] Add ``JARVISDataset`` . There is already a (yet not fully) port for `kgcnn` .
* [ ] Add package wide Logger Level to change. 
* [x] Training scripts need all seed for maximum reproducibility.