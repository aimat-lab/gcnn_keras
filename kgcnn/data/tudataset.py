import numpy as np
import os

from kgcnn.data.base import GraphDatasetBase


class GraphTUDataset(GraphDatasetBase):

    all_tudataset_identifier = ["PROTEINS", "MUTAG", "Mutagenicity"]

    def __init__(self, tudataset_name=None, reload=False, verbose=1):
        if isinstance(tudataset_name, str) and tudataset_name in self.all_tudataset_identifier:
            self.data_directory = tudataset_name
            self.download_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/"+tudataset_name+".zip"
            self.file_name = tudataset_name+".zip"
            self.unpack_zip = True
            self.unpack_directory = tudataset_name
            self.fits_in_memory = True
            self.kgcnn_dataset_name = tudataset_name

        super(GraphTUDataset, self).__init__(reload=reload, verbose=verbose)

    def read_in_memory(self, verbose=1):

        if self.file_name is not None and self.kgcnn_dataset_name in self.all_tudataset_identifier:
            name_dataset = self.kgcnn_dataset_name
            path = os.path.join(self.data_main_dir, self.data_directory, self.unpack_directory, name_dataset)
        else:
            print("WARNING: Dataset with name", self.kgcnn_dataset_name, "not found in TUDatasets list.")
            return None

        # Define a graph with indices
        # They must be defined
        g_a = np.array(self.read_csv_simple(os.path.join(path, name_dataset + "_A.txt"), dtype=int), dtype="int")
        g_n_id = np.array(self.read_csv_simple(os.path.join(path, name_dataset + "_graph_indicator.txt"), dtype=int),
                          dtype="int")

        # Try read in labels and attributes (optional)
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
        num_graphs = len(g_labels)

        # shift index, should start at 0 for python indexing
        if int(np.amin(g_n_id)) == 1 and int(np.amin(g_a)) == 1:
            if verbose > 0:
                print("INFO: Shift index of graph id to zero for", name_dataset, "to match python indexing.")
            g_a = g_a - 1
            g_n_id = g_n_id - 1

        # split into separate graphs
        graph_id, counts = np.unique(g_n_id, return_counts=True)
        graphlen = np.zeros(num_graphs, dtype=np.int)
        graphlen[graph_id] = counts

        if n_attr is not None:
            n_attr = np.split(n_attr, np.cumsum(graphlen)[:-1])
        if n_labels is not None:
            n_labels = np.split(n_labels, np.cumsum(graphlen)[:-1])

        # edge_indicator
        graph_id_edge = g_n_id[g_a[:, 0]]  # is the same for adj_matrix[:,1]
        graph_id2, counts_edge = np.unique(graph_id_edge, return_counts=True)
        edgelen = np.zeros(num_graphs, dtype=np.int)
        edgelen[graph_id2] = counts_edge

        if e_attr is not None:
            e_attr = np.split(e_attr, np.cumsum(edgelen)[:-1])
        if e_labels is not None:
            e_labels = np.split(e_labels, np.cumsum(edgelen)[:-1])

        # edge_indices
        node_index = np.concatenate([np.arange(x) for x in graphlen], axis=0)
        edge_indices = node_index[g_a]
        edge_indices = np.concatenate([edge_indices[:,1:], edge_indices[:,:1]],axis=-1)  # switch indices
        edge_indices = np.split(edge_indices, np.cumsum(edgelen)[:-1])

        # Check if unconnected
        all_cons = []
        for i in range(num_graphs):
            cons = np.arange(graphlen[i])
            test_cons = np.sort(np.unique(cons[edge_indices[i]].flatten()))
            is_cons = np.zeros_like(cons, dtype=np.bool)
            is_cons[test_cons] = True
            all_cons.append(np.sum(is_cons == False))
        all_cons = np.array(all_cons)

        if verbose > 0:
            print("INFO: Graph index which has unconnected", np.arange(len(all_cons))[all_cons > 0], "with",
                  all_cons[all_cons > 0], "in total", len(all_cons[all_cons > 0]))

        node_degree = [np.zeros(x, dtype="int") for x in graphlen]
        for i, x in enumerate(edge_indices):
            nod_id, nod_counts = np.unique(x[:, 0], return_counts=True)
            node_degree[i][nod_id] = nod_counts

        self.nodes_degree = node_degree
        self.nodes = n_attr
        self.edges = e_attr
        self.graph_state = g_attr
        self.edge_indices = edge_indices
        self.labels_node = n_labels
        self.labels_edge = e_labels
        self.labels_graph = g_labels

    def get_graph(self):
        """Make vanilla graph tensor objects.

        Returns:
            tuple: labels, nodes, edge_indices, edges
        """
        return self.labels_graph, self.nodes, self.edge_indices, self.edges
