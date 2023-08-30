v3.1.0

* Added flexible charge for ``rdkit_xyz_to_mol`` as e.g. list.
* Added ``from_xyz`` to ``MolecularGraphRDKit`` .
* Started additional ``kgcnn.molecule.preprocessor`` module for graph preprocessors.
* BREAKING CHANGES: Renamed module ``kgcnn.layers.pooling`` to ``kgcnn.layers.aggr`` for better compatibility.
However, kept legacy pooling module and all old ALIAS.
* Repair bug in ``RelationalMLP`` .
* `HyperParameter` is not verified on initialize anymore, just call `hyper.verify()`.
* Moved losses from `kgcnn.metrics.loss` into separate modul ``kgcnn.losses`` to be more compatible with keras.
* Reworked training scripts especially to simplify command line arguments and strengthen hyperparameter.
* Started with potential keras-core port. Not yet tested or supported. 


v3.0.2

* Added ``add_eps`` to `PAiNNUpdate` layer as option.
* Rework ``data.transform.scaler.standard`` to hopefully now fix all errors with the scalers.
* BREAKING CHANGES: Refactored activation functions `kgcnn.ops.activ` and layers `kgcnn.layers.activ` that have trainable parameters, due to keras changes in 2.13.0. 
  Please check your config, since parameters are ignored in normal functions!
  If for example "kgcnn>leaky_relu" you can not change the leak anymore. You must use a ``kgcnn.layers.activ`` for that.
* Rework ``kgcnn.graph.methods.range_neighbour_lattice`` to use pymatgen.
* Added ``PolynomialDecayScheduler``
* Added option for force model to use normal gradient and added as option ``use_batch_jacobian`` .
* BREAKING CHANGES: Reworked `kgcnn.layers.gather` to reduce/simplify code and speed up some models. 
  The behaviour of `GatherNodes` has changed a little in that it first splits and then concatenates. The default parameters now have `split_axis` and `concat_axis` set to 2. `concat_indices` has been removed.
  The default behaviour of the layer however stays the same.
* An error in layer `FracToRealCoordinates` has been fixed and improved speed.
* Removed deprecated ``kgcnn.model.utils.generate_embedding`` .
* 


v3.0.1

* Removed deprecated molecules.
* Fix error in ``kgcnn.data.transform.scaler.serial``
* Fix error in ``QMDataset`` if attributes have been chosen. Now `set_attributes` does not cause an error.
* Fix error in ``QMDataset`` with labels without SDF file.
* Fix error in ``kgcnn.layers.conv.GraphSageNodeLayer`` .
* Add ``reverse_edge_indices`` option to `GraphDict.from_networkx` . Fixed error in connection with `kgcnn.crystal` .
* Started with ``kgcnn.io.file`` . Experimental. Will get more updates.
* Fix error with `StandardLabelScaler` inheritance.
* Added workflow notebook examples. 
* Fix error in import ``kgcnn.crystal.periodic_table`` to now properly include package data.


v3.0.0

Major refactoring of kgcnn layers and models. 
We try to provide the most important layers for graph convolution as ``kgcnn.layers`` with ragged tensor representation.
As for literature models only input and output is matched with ``kgcnn`` .

* Move ``kgcnn.layers.conv`` to `kgcnn.literature` .
* Refactored all graph methods in ``graph.methods`` .
* Moved ``kgcnn.mol.*`` and `kgcnn.moldyn.*` into `kgcnn.molecule`
* Moved ``hyper`` into `trainig`
* Updated ``crystal`` .


v2.2.4

* Added ``ACSFConstNormalization`` to literature models as option.
* Adjusted and reworked ``MLP`` . Now includes more normalization options. 
* Removed 'is_sorted', 'node_indexing' and 'has_unconnected' from ``GraphBaseLayer`` and added it to the pooling layers directly.


v2.2.3

* HOTFIX: Changed ``MemoryGraphList.tensor()`` so that the correct dtype is given to the tensor output. This is important for model loading etc.
* Added ``CENTChargePlusElectrostaticEnergy`` to `kgcnn.layers.conv.hdnnp_conv` and `kgcnn.literature.HDNNP4th` .
* Fix bug in latest ``train_force.py`` of v2.2.2 that forgets to apply inverse scaling to dataset, causing subsequent folds to have wrong labels.
* HOTFIX: Updated ``MolDynamicsModelPredictor`` to call keras model without very expensive retracing. Alternative mode use `use_predict=True` .
* Update training results and data subclasses for matbench datasets.
* Added ``GraphInstanceNormalization`` and `GraphNormalization` to `kgcnn.layers.norm` .


v2.2.2

* Reworked all scaler class to have separate name for using either X or y. For example ``StandardScaler`` or ``StandardLabelScaler`` .
* Moved scalers to ``kgcnn.data.transform`` . We will expand on this in the future.
* IMPORTANT: Renamed and changed behaviour for ``EnergyForceExtensiveScaler`` . New name is `EnergyForceExtensiveLabelScaler` . Return is just y now. Added experimental functionality for transforming dataset.
* Adjusted training scripts for new scalers.
* Reduced requirements for tensorflow to 2.9.
* Renamed ``kgcnn.md`` to `kgcnn.moldyn` for naming conflicts with markdown.
* In ``MolDynamicsModelPredictor`` renamed argument `model_postprocessor` to `graph_postprocessor` .


v2.2.1

* HOTFIX: Removed ``tensorflow_gpu`` from setup.py
* Added ``HDNNP4th.py`` to literature.
* Fixed error in ``ChangeTensorType`` config for model save.
* Merged pull request for #103 for ``kgcnn.xai`` .


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
* Added further documentation.


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