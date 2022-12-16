v2.2.0

* Removed deprecated modules in ``kgcnn.utils``.
* Moved ``kgcnn.utils.model`` to `kgcnn.model.utils`.
* Fixed behaviour for ``kgcnn.data.base.MemoryGraphDataset.get_train_test_indices`` to return list of train-test index tuples.
* Updated ``kgcnn.hyper.hyper.HyperParameter`` to deserialize metrics and loss for multi-output models.
* Added `trajectory_name` in `summary.py` and `history.py`. 
* Fixed ``kgcnn.layers.geom.PositionEncodingBasisLayer``
* Removed deprecated ``kgcnn.layers.conv.attention`` and ``kgcnn.layers.conv.message``
* Updated ``setup.py`` for requirements.
* HOTFIX: Error in ``kgcnn.scaler.mol.ExtensiveMolecuarScaler``, where scale was not properly applied. However, this was only present in development (master) version, not in release.
* Added ``kgcnn.layers.relational`` with relation dense layer.
* Added first draft of ``kgcnn.literature.HDNNP2nd``.
* Added ``append``, `update` and `add` to `MemoryGraphList`.
* Fixed behaviour of ``GraphDict`` , which now is not making a copy of arrays.
* Added explainable GNN from ``visual_graph_datasets``.
* Ran training for ``train_force.py``
* Changed backend to RDkit for ``QMDatasets``.
* Added ``kgcnn.md`` and ``kgcnn.xai`` .


v2.1.1

* Removed `kgcnn.graph.adapter` and switched to completed ``kgcnn.graph.preprocessor``. The interface to `MemoryGraphList` and datasets does not change. How to update:
```python
from kgcnn.data.base import GraphDict
GraphDict().apply_preprocessor("sort_edge_indices")  # Instead of GraphDict().sort_edge_indices()
GraphDict().apply_preprocessor("set_edge_weights_uniform", value=0.0) # Instead of GraphDict().set_edge_weights_uniform(value=0.0)
# Or directly using class.
from kgcnn.graph.preprocessor import SortEdgeIndices
SortEdgeIndices(in_place=True)(GraphDict())
```
* Add ``kgcnn.literature.MEGAN`` model.
* Add ``kgcnn.literature.MXMNet`` model.
* Fixed error in ``ClinToxDataset`` label index.
* Reworked ``kgcnn.graph.adj.get_angle_index`` with additional function arguments. Default behaviour remains identical. For periodic system an additional `allow_reverse_edges=True` is now required.
* Added input embedding for edges in ``kgcnn.literature.EGNN``. Debugged model.
* Reworked ``kgcnn.literature.PAiNN`` to simplify normalization option and add equivariant initialization method.
* Refactored ``kgcnn.data.qm`` including all qm7-qm9 datasets. Improved documentation. If error occurs, please run `QM9Dataset(reload=True)`.
* Refactored ``kgcnn.data.moleculenet``. Interface and behaviour does not change.
* Renamed ``kgcnn.mol.graph_babel`` and ``kgcnn.mol.graph_rdkit`` and move conversion into ``kgcnn.mol.convert``.
* Added ``kgcnn.data.datasets.MD17Dataset`` and ``kgcnn.data.datasets.MD17RevisedDataset``
* Refactored ``kgcnn.scaler`` module to follow sklearn definitions. Changed input naming and order for scaler. Add config and weights functionality.
* Changed ``kgcnn.training.scheduler.LinearWarmupExponentialLearningRateScheduler`` to take correctly lifetime parameter. 
* Reworked ``kgcnn.data.datasets.QM9Dataset`` to offer atomization energies and uncharacterized molecules. Please run reload=True.
* Improved docs for `kgcnn.training`.


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