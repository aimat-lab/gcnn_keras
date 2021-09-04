import numpy as np
import os

from kgcnn.data.base import DownloadDataset, MemoryGraphDataset


# TUDataset: A collection of benchmark datasets for learning with graphs
# by Christopher Morris and Nils M. Kriege and Franka Bause and Kristian Kersting and Petra Mutzel and Marion Neumann
# http://graphlearning.io


class GraphTUDataset(DownloadDataset, MemoryGraphDataset):
    r"""Base class for loading graph datasets published by `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_. Datasets contain non-isomorphic graphs. This general base class has
    functionality to load TUDatasets in a generic way.

    .. note::
        Note that sub-classes of `GraphTUDataset` in :obj:``kgcnn.data.datasets`` should still be made,
        if the dataset needs more refined post-precessing. Not all datasets can provide all types of graph
        properties like `edge_attributes` etc.

    """

    # List of datasets in TUDatasets.
    tudataset_ids = [
        # Molecules
        "AIDS", "alchemy_full", "aspirin", "benzene", "BZR", "BZR_MD", "COX2", "COX2_MD", "DHFR", "DHFR_MD", "ER_MD",
        "ethanol", "FRANKENSTEIN", "malonaldehyde", "MCF-7", "MCF-7H", "MOLT-4", "MOLT-4H", "Mutagenicity", "MUTAG",
        "naphthalene", "NCI1", "NCI109", "NCI-H23", "NCI-H23H", "OVCAR-8", "OVCAR-8H", "P388", "P388H", "PC-3", "PC-3H",
        "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "QM9", "salicylic_acid", "SF-295", "SF-295H", "SN12C", "SN12CH",
        "SW-620", "SW-620H", "toluene", "Tox21_AhR_training", "Tox21_AhR_testing", "Tox21_AhR_evaluation",
        "Tox21_AR_training", "Tox21_AR_testing", "Tox21_AR_evaluation", "Tox21_AR-LBD_training", "Tox21_AR-LBD_testing",
        "Tox21_AR-LBD_evaluation", "Tox21_ARE_training", "Tox21_ARE_testing", "Tox21_ARE_evaluation",
        "Tox21_aromatase_training", "Tox21_aromatase_testing", "Tox21_aromatase_evaluation", "Tox21_ATAD5_training",
        "Tox21_ATAD5_testing", "Tox21_ATAD5_evaluation", "Tox21_ER_training", "Tox21_ER_testing", "Tox21_ER_evaluation",
        "Tox21_ER-LBD_training", "Tox21_ER-LBD_testing", "Tox21_ER-LBD_evaluation", "Tox21_HSE_training",
        "Tox21_HSE_testing", "Tox21_HSE_evaluation", "Tox21_MMP_training", "Tox21_MMP_testing", "Tox21_MMP_evaluation",
        "Tox21_p53_training", "Tox21_p53_testing", "Tox21_p53_evaluation", "Tox21_PPAR-gamma_training",
        "Tox21_PPAR-gamma_testing", "Tox21_PPAR-gamma_evaluation", "UACC257", "UACC257H", "uracil", "Yeast", "YeastH",
        "ZINC_full", "ZINC_test", "ZINC_train", "ZINC_val",
        # Bioinformatics
        "DD", "ENZYMES", "KKI", "OHSU", "Peking_1", "PROTEINS", "PROTEINS_full",
        # Computer vision
        "COIL-DEL", "COIL-RAG", "Cuneiform", "Fingerprint", "FIRSTMM_DB", "Letter-high", "Letter-low", "Letter-med",
        "MSRC_9", "MSRC_21", "MSRC_21C",
        # Social networks
        "COLLAB", "dblp_ct1", "dblp_ct2", "DBLP_v1", "deezer_ego_nets", "facebook_ct1", "facebook_ct2",
        "github_stargazers", "highschool_ct1", "highschool_ct2", "IMDB-BINARY", "IMDB-MULTI", "infectious_ct1",
        "infectious_ct2", "mit_ct1", "mit_ct2", "REDDIT-BINARY", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K",
        "reddit_threads", "tumblr_ct1", "tumblr_ct2", "twitch_egos", "TWITTER-Real-Graph-Partial",
        # Synthetic
        "COLORS-3", "SYNTHETIC", "SYNTHETICnew", "Synthie", "TRIANGLES"
    ]

    def __init__(self, dataset_name: str, reload: bool = False, verbose: int = 1):
        """Initialize a `GraphTUDataset` instance from string identifier.

        Args:
            dataset_name (str): Name of a dataset.
            reload (bool): Download the dataset again and prepare data on disk.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        if not isinstance(dataset_name, str):
            raise ValueError("ERROR:kgcnn: Please provide string identifier for TUDataset.")

        if dataset_name in self.tudataset_ids:
            self.data_directory = dataset_name
            self.download_url = "https://www.chrsmrrs.com/graphkerneldatasets/"
            self.download_url = self.download_url + dataset_name + ".zip"
            self.file_name = dataset_name + ".zip"
            self.unpack_zip = True
            self.unpack_directory = dataset_name
            self.fits_in_memory = True
            self.dataset_name = dataset_name
        else:
            print("ERROR:kgcnn: Can not resolve %s as a TUDataset." % dataset_name,
                  "Add to `all_tudataset_identifier` list manually.")

        DownloadDataset.__init__(self, reload=reload, verbose=verbose)
        MemoryGraphDataset.__init__(self, verbose=verbose)
        if verbose > 1:
            print("INFO:kgcnn: Reading dataset to memory with name %s" % str(self.dataset_name))

        if self.fits_in_memory:
            self.read_in_memory(verbose=verbose)

    def read_in_memory(self, verbose: int = 1):
        r"""Read the TUDataset into memory. The TUDataset is stored in disjoint representations. The data is cast
        to a list of separate graph properties for `MemoryGraphDataset`.

        Args:
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.

        Returns:
            self
        """

        if self.file_name is not None and self.dataset_name in self.tudataset_ids:
            name_dataset = self.dataset_name
            path = os.path.join(self.data_main_dir, self.data_directory, self.unpack_directory, name_dataset)
        else:
            print("WARNING:kgcnn: Dataset with name %s not found in TUDatasets list." % self.dataset_name)
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
        num_graphs = np.amax(g_n_id)
        if g_labels is not None:
            if len(g_labels) != num_graphs:
                print("ERROR:kgcnn: Wrong number of graphs, not matching graph labels, {0}, {1}".format(len(g_labels),
                                                                                                        num_graphs))

        # shift index, should start at 0 for python indexing
        if int(np.amin(g_n_id)) == 1 and int(np.amin(g_a)) == 1:
            if verbose > 0:
                print("INFO:kgcnn: Shift start of graph id to zero for %s to match python indexing." % name_dataset)
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
        edge_indices = np.concatenate([edge_indices[:, 1:], edge_indices[:, :1]], axis=-1)  # switch indices
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
            print("INFO:kgcnn: Graph index which has unconnected", np.arange(len(all_cons))[all_cons > 0], "with",
                  all_cons[all_cons > 0], "in total", len(all_cons[all_cons > 0]))

        node_degree = [np.zeros(x, dtype="int") for x in graphlen]
        for i, x in enumerate(edge_indices):
            nod_id, nod_counts = np.unique(x[:, 0], return_counts=True)
            node_degree[i][nod_id] = nod_counts

        self.node_degree = node_degree
        self.node_attributes = n_attr
        self.edge_attributes = e_attr
        self.graph_attributes = g_attr
        self.edge_indices = edge_indices
        self.node_labels = n_labels
        self.edge_labels = e_labels
        self.graph_labels = g_labels
        self.length = num_graphs

        return self

    @staticmethod
    def _debug_read_list():
        line_ids = []
        with open("datasets.md", 'r') as f:
            for line in f.readlines():
                if line[:3] == "|**":
                    line_ids.append(line.split("**")[1])
        return line_ids

