v2.1.1


v2.1.0

* Remove reserved properties form ``MemoryGraphList``, please use set/get methods.
* Removed deprecated ``kgcnn.selection`` module.
* Added history score summary in ``kgcnn.training.history``.
* Rework training. Having plots takes up more memory. Prefer summary table of benchmarks.
* Changed ``kgcnn.data.datasets.PROTEINSDataset`` to binary graph labels.
* Add ``kgcnn.literature.CMPNN`` model.
* Add ``kgcnn.literature.EGNN`` model.
* Merge ``set_attributes`` into `read_in_memory` for `MoleculeNetDataset` and make ``set_attributes`` alias of `read_in_memory`.   
* Fix error of node updates in ``kgcnn.literature.GAT``. Rerunning training.
* Fix learning rate scheduler ``kgcnn.training.scheduler.LinearLearningRateScheduler`` min. learning rate if trained beyond epoch argument.
* Removed ``kgcnn.layers.casting.ChangeIndexing`` at it was not used.
* Added ``kgcnn.layers.casting.CastEdgeIndicesToDenseAdjacency``.
* Merged ``kgcnn.layers.mlp.MLP`` and ``kgcnn.layers.mlp.GraphMLP``, but kept `GraphMLP` as alias. Change in kwargs for "normalization_technique".
* Moved ``kgcnn.layers.conv.message`` to ``kgcnn.layers.message``.
* Refactored ``kgcnn.layers.conv.attention`` into ``kgcnn.layers.conv.gat_conv`` and ``kgcnn.layers.conv.attentivefp_conv``.
* In ``MoleculeNetDataset`` and `QMDataset` changed the shape of 'edge_number' to be `(N, )` instead of `(N, 1)`. To agree with 'node_number' shape.
* Removed ``kgcnn.layers.conv.sparse`` as it was not used and added its content to ``kgcnn.layers.conv.gcn_conv`` and ``kgcnn.layers.casting`` 
* Started with ``kgcnn.graph.preprocessor``.

v2.0.4

* Add ``kgcnn.crystal`` module, which is still in development.
* Add ``get_weights`` and ``get_config`` to `kgcnn.scaler` 
* Add ``get`` and ``set`` alias to `GraphDict` and `MemoryGraphList`. Which now can be used to assign and obtain graph properties.
* Refactored ``GraphDict`` and `adj` into `kgcnn.graph`.
* Add a ``set_range_periodic`` function to `GraphDict`.
* Add ``make_crystal_model`` functions to SchNet, Megnet, DimeNetPP.
* Add ``custom_transform`` to `MoleculeNetDataset`.
* Removed ``add_hydrogen``, `make_conformer`, and `optimize_conformer` from constructor of `MolGraphInterface`.
* Added ``add_hs``, `make_conformer` and `optimize_conformer` to `MolGraphInterface`.
* Add normalization option to PAiNN and add ``make_crystal_model``.
* Add ``kgcnn.literature.CGCNN`` model with docs.
* Add more list-functionality to ``MemoryGraphList``.
* Add _tutorial_model_loading_options.ipynb_ to _notebooks_ showing different ways to load ragged data.
* Add _tutorial_hyper_optuna.ipynb_ to _notebooks_


v2.0.3

* fix typo to read `kgcnn.mol.encoder`
* fix bug in ``GraphDict.from_networkx()`` for edge attributes.
* Improved docs overall.
* Added ragged node/edge embedding output for TF > 2.8 via "output_to_tensor" model config.
* Added ``make_function`` option to training scripts.
* Refactored GraphDict methods into ``kgcnn.data.adapter.GraphTensorMethodsAdapter``.
* Removed ``kgcnn.layers.modules.ReduceSum`` as it has not been used and may be problematic.
* Moved ``kgcnn.utils.data`` to ``kgcnn.data.utils``. 
* Refactored smile to mol generation into ``kgcnn.mol.convert`` and renamed `kgcnn.mol.gen` to `kgcnn.mol.external`
* fixed bug for `GatherEmbedding` to have correct concat axis if index tensor happens to be of rank>3 but ragged_rank=1.
* Refactored `kgcnn.mol` methods into modules and renamed `graphRD` and `graphBabel`.
* Continued to work on ``kgcnn.data.crystal.CrystalDataset``.
* Added ``MatBenchDataset2020`` dataset to `kgcnn.data.datasets`.