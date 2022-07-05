v2.0.3

* fix typo to read `kgcnn.mol.encoder`
* fix bug in ``GraphDict.from_networkx()`` for edge attributes.
* Improved docs overall.
* Added ragged node/edge embedding output for TF > 2.8 via "output_to_tensor" model config.
* Added ``make_function`` option to training scripts.
* Refactored GraphDict methods into ``kgcnn.data.adapter.GraphMethodsAdapter``.
* Removed ``kgcnn.layers.modules.ReduceSum`` as it has not been used and may be problematic.
* Moved ``kgcnn.utils.data`` to ``kgcnn.data.utils``. 
* Refactored smile to mol generation into ``kgcnn.mol.convert`` and renamed `kgcnn.mol.gen` to `kgcnn.mol.external`
* fixed bug for `GatherEmbedding` to have correct concat axis if index tensor happens to be of rank>3 but ragged_rank=1.
* Refactored `kgcnn.mol` methods into modules and renamed `graphRD` and `graphBabel`.
* Continued to work on ``kgcnn.data.crystal.CrystalDataset``.
* Added ``MatBenchDataset2020`` dataset to `kgcnn.data.datasets`.