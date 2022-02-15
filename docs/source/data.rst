.. _data:
   :maxdepth: 3

Datasets
========


How to construct ragged tensors is shown above.
Moreover, some data handling classes are given in ``kgcnn.data``.
Graphs are represented by a dictionary of (numpy) tensors ``GraphDict`` and are stored in a list ``MemoryGraphList``.
Both must fit into memory and are supposed to be handled just like a python dict or list, respectively.::

    from kgcnn.data.base import GraphDict, MemoryGraphList
    # Single graph.
    graph = GraphDict({"edge_indices": [[0, 1], [1, 0]]})
    print(graph)
    # List of graph dicts.
    graph_list = MemoryGraphList([graph, {"edge_indices": [[0, 0]]}, {}])
    graph_list.clean(["edge_indices"])  # Remove graphs without property
    graph_list.obtain_property("edge_indices")  # opposite is assign_property()
    graph_list.tensor([{"name": "edge_indices", "ragged": True}]) # config of layers.Input; makes copy.


The ``MemoryGraphDataset`` inherits from ``MemoryGraphList`` but must be initialized with file information on disk that points to a ``data_directory`` for the dataset.
The ``data_directory`` can have a subdirectory for files and/or single file such as a CSV file::

    ├── data_directory
        ├── file_directory
        │   ├── *.*
        │   └── ...
        ├── file_name
        └── dataset_name.pickle

A base dataset class is created with path and name information::

    from kgcnn.data.base import MemoryGraphDataset
    dataset = MemoryGraphDataset(data_directory="ExampleDir/",
                                 dataset_name="Example",
                                 file_name=None, file_directory=None)


The subclasses ``QMDataset``, ``MoleculeNetDataset`` and ``GraphTUDataset`` further have functions required for the specific dataset type to convert and process files such as '.txt', '.sdf', '.xyz' etc.
Most subclasses implement ``prepare_data()`` and ``read_in_memory()`` with dataset dependent arguments.
An example for ``MoleculeNetDataset`` is shown below.
For mote details find tutorials in `notebooks <https://github.com/aimat-lab/gcnn_keras/tree/master/notebooks>`_.::

    from kgcnn.data.moleculenet import MoleculeNetDataset
    # File directory and files must exist.
    # Here 'ExampleDir' and 'ExampleDir/data.csv' with columns "smiles" and "label".
    dataset = MoleculeNetDataset(dataset_name="Example",
                                 data_directory="ExampleDir/",
                                 file_name="data.csv")
    dataset.prepare_data(overwrite=True, smiles_column_name="smiles", add_hydrogen=True,
                         make_conformers=True, optimize_conformer=True, num_workers=None)
    dataset.read_in_memory(label_column_name="label",  add_hydrogen=False,
                           has_conformers=True)


In `data.datasets <https://github.com/aimat-lab/gcnn_keras/tree/master/kgcnn/data/datasets>`_ there are graph learning benchmark datasets as subclasses which are being *downloaded* from e.g. popular graph archives like `TUDatasets <https://chrsmrrs.github.io/datasets/>`_ or `MoleculeNet <https://moleculenet.org/>`_.
The subclasses ``GraphTUDataset2020`` and ``MoleculeNetDataset2018`` download and read the available datasets by name.
There are also specific dataset subclass for each dataset to handle additional processing or downloading from individual sources::


    from kgcnn.data.datasets.MUTAGDataset import MUTAGDataset
    dataset = MUTAGDataset()  # inherits from GraphTUDataset2020


Downloaded datasets are stored in "/.kgcnn/datasets" on your computer. Please remove them manually, if no longer required.
