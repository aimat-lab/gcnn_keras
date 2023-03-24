import numpy as np
import os
from typing import Callable
# import logging

from kgcnn.data.base import MemoryGraphDataset

# TUDataset: A collection of benchmark datasets for learning with graphs
# by Christopher Morris and Nils M. Kriege and Franka Bause and Kristian Kersting and Petra Mutzel and Marion Neumann
# http://graphlearning.io


class GraphTUDataset(MemoryGraphDataset):
    r"""Base class for loading graph datasets published by `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.

    Datasets contain non-isomorphic graphs for many graph classification or regression tasks.
    This general base class has functionality to load TUDatasets in a generic way.
    The datasets are already in a graph-like format and do not need further processing via e.g. `prepare data`.

    .. note::

        Note that subclasses of `GraphTUDataset2020` in :obj:`kgcnn.data.datasets` downloads datasets,
        There are also further TU-datasets in :obj:`kgcnn.data.datasets`, if further processing is used in literature.
        Not all datasets can provide all types of graph properties like `edge_attributes` etc.

    The file structure of :obj:`GraphTUDataset` for a given dataset 'DS' (replace DS with dataset name).

    .. code-block:: console

        ├── data_directory
            ├── DS_graph_indicator.txt
            ├── DS_A.txt
            ├── DS_node_labels.txt
            ├── DS_node_attributes.txt
            ├── DS_edge_labels.txt
            ├── DS_edge_attributes.txt
            ├── DS_graph_labels.txt
            ├── DS_graph_attributes.txt
            ├──  ...
            └── dataset_name.kgcnn.pickle

    Setting up a single file can be made additionally with base class :obj:`save` method.
    """

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 file_directory: str = None, verbose: int = 10):
        r"""Initialize a :obj:`GraphTUDataset` instance from file.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Filename for reading into memory. Not used for general TUDataset, since there are multiple
                files with a prefix and pre-defined suffix. Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                files. Default is None.
            dataset_name (str): Name of the dataset. Important for base-name for naming of files. Default is None.
            verbose (int): Logging level. Default is 10.
        """
        MemoryGraphDataset.__init__(self, data_directory=data_directory, dataset_name=dataset_name,
                                    file_name=file_name, verbose=verbose, file_directory=file_directory)

    def read_in_memory(self):
        r"""Read the :obj:`GraphTUDataset` into memory.

        The TUDataset is stored in disjoint representations. The data is cast
        to a list of separate graph properties for `MemoryGraphDataset`.

        Returns:
            self
        """
        if self.dataset_name is not None and self.data_directory is not None:
            path = os.path.realpath(self.data_directory)
            name_dataset = self.dataset_name
            if self.file_directory is not None:
                path = os.path.join(path, self.file_directory)
        else:
            self.error("Dataset needs name {0} and path {1}.".format(self.dataset_name, self.data_directory))
            return self

        self.info("Reading dataset to memory with name %s" % str(self.dataset_name))

        # Define a graph with indices
        # They must be defined
        g_a = np.array(self.read_csv_simple(os.path.join(path, name_dataset + "_A.txt"), dtype=int), dtype="int")
        g_n_id = np.array(self.read_csv_simple(os.path.join(path, name_dataset + "_graph_indicator.txt"), dtype=int),
                          dtype="int")

        # Try read in labels.
        try:
            g_labels = np.array(
                self.read_csv_simple(os.path.join(path, name_dataset + "_graph_labels.txt"), dtype=float))
        except FileNotFoundError:
            g_labels = None
        try:
            n_labels = np.array(
                self.read_csv_simple(os.path.join(path, name_dataset + "_node_labels.txt"), dtype=float))
        except FileNotFoundError:
            n_labels = None
        try:
            e_labels = np.array(
                self.read_csv_simple(os.path.join(path, name_dataset + "_edge_labels.txt"), dtype=float))
        except FileNotFoundError:
            e_labels = None

        # Try read in attributes
        try:
            n_attr = np.array(
                self.read_csv_simple(os.path.join(path, name_dataset + "_node_attributes.txt"), dtype=float))
        except FileNotFoundError:
            n_attr = None
        try:
            e_attr = np.array(
                self.read_csv_simple(os.path.join(path, name_dataset + "_edge_attributes.txt"), dtype=float))
        except FileNotFoundError:
            e_attr = None
        try:
            g_attr = np.array(
                self.read_csv_simple(os.path.join(path, name_dataset + "_graph_attributes.txt"), dtype=float))
        except FileNotFoundError:
            g_attr = None

        # labels
        num_graphs = np.amax(g_n_id)
        if g_labels is not None:
            if len(g_labels) != num_graphs:
                self.error(
                    "Wrong number of graphs, not matching graph labels, {0}, {1}.".format(len(g_labels), num_graphs))

        # shift index, should start at 0 for python indexing
        if int(np.amin(g_n_id)) == 1 and int(np.amin(g_a)) == 1:
            self.info("Shift start of graph ID to zero for '%s' to match python indexing." % name_dataset)
            g_a = g_a - 1
            g_n_id = g_n_id - 1

        # split into separate graphs
        graph_id, counts = np.unique(g_n_id, return_counts=True)
        graph_len = np.zeros(num_graphs, dtype="int")
        graph_len[graph_id] = counts

        if n_attr is not None:
            n_attr = np.split(n_attr, np.cumsum(graph_len)[:-1])
        if n_labels is not None:
            n_labels = np.split(n_labels, np.cumsum(graph_len)[:-1])

        # edge_indicator
        graph_id_edge = g_n_id[g_a[:, 0]]  # is the same for adj_matrix[:,1]
        graph_id2, counts_edge = np.unique(graph_id_edge, return_counts=True)
        edge_len = np.zeros(num_graphs, dtype="int")
        edge_len[graph_id2] = counts_edge

        if e_attr is not None:
            e_attr = np.split(e_attr, np.cumsum(edge_len)[:-1])
        if e_labels is not None:
            e_labels = np.split(e_labels, np.cumsum(edge_len)[:-1])

        # edge_indices
        node_index = np.concatenate([np.arange(x) for x in graph_len], axis=0)
        edge_indices = node_index[g_a]
        edge_indices = np.concatenate([edge_indices[:, 1:], edge_indices[:, :1]], axis=-1)  # switch indices
        edge_indices = np.split(edge_indices, np.cumsum(edge_len)[:-1])

        # Check if unconnected
        all_cons = []
        for i in range(num_graphs):
            cons = np.arange(graph_len[i])
            test_cons = np.sort(np.unique(cons[edge_indices[i]].flatten()))
            is_cons = np.zeros_like(cons, dtype="bool")
            is_cons[test_cons] = True
            all_cons.append(np.sum(np.invert(is_cons)))
        all_cons = np.array(all_cons)

        self.info("Graph index which has unconnected '%s' with '%s' in total '%s'." % (
            np.arange(len(all_cons))[all_cons > 0], all_cons[all_cons > 0], len(all_cons[all_cons > 0])))

        node_degree = [np.zeros(x, dtype="int") for x in graph_len]
        for i, x in enumerate(edge_indices):
            nod_id, nod_counts = np.unique(x[:, 0], return_counts=True)
            node_degree[i][nod_id] = nod_counts

        # Assert list for graph items.
        g_attr = [x for x in g_attr] if g_attr is not None else None
        g_labels = [x for x in g_labels] if g_labels is not None else None

        # Assign to self.
        for key, value in {"node_degree": node_degree, "node_attributes": n_attr, "node_labels": n_labels,
                           "edge_attributes": e_attr, "edge_indices": edge_indices, "edge_labels": e_labels,
                           "graph_attributes": g_attr, "graph_labels": g_labels}.items():
            self.assign_property(key, value)

        return self

    @staticmethod
    def read_csv_simple(filepath: str, delimiter: str = ",", dtype: Callable = float):
        """Very simple python-only function to read in a csv-file from file.

        Args:
            filepath (str): Full filepath of csv-file to read in.
            delimiter (str): Delimiter character for separation. Default is ",".
            dtype: Callable type conversion from string. Default is float.

        Returns:
            list: Python list of values. Length of the list equals the number of lines.
        """
        out = []
        open_file = open(filepath, "r")
        for lines in open_file.readlines():
            string_list = lines.strip().split(delimiter)
            values_list = [dtype(x.strip()) for x in string_list]
            out.append(values_list)
        open_file.close()
        return out
