import os
import numpy as np

from kgcnn.data.base import GraphDatasetBase


class CoraLUDataset(GraphDatasetBase):
    """Store and process Cora dataset after Lu et al. 2003."""

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "cora_lu"
    download_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    # download_url = "https://linqs-data.soe.ucsc.edu/public/arxiv-mrdm05/arxiv.tar.gz"
    file_name = 'cora.tgz'
    unpack_tar = True
    unpack_zip = False
    unpack_directory = "cora_lu"
    fits_in_memory = True

    # Make cora graph that was published by Qing Lu, and Lise Getoor. "Link-based classification." ICML, 2003.
    # https://www.aaai.org/Papers/ICML/2003/ICML03-066.pdf

    def __init__(self, reload=False, verbose=1):
        """Initialize Cora dataset after Lu et al. 2003.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        self.class_label_mapping = None
        # Use default base class init()
        super(CoraLUDataset, self).__init__(reload=reload, verbose=verbose)

    def read_in_memory(self, verbose=1):
        """Load Cora data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        filepath = os.path.join(self.data_main_dir, self.data_directory, self.unpack_directory, "cora")

        ids = np.loadtxt(os.path.join(filepath, "cora.cites"))
        ids = np.array(ids, np.int)
        open_file = open(os.path.join(filepath, "cora.content"), "r")
        lines = open_file.readlines()
        labels = [x.strip().split('\t')[-1] for x in lines]
        nodes = [x.strip().split('\t')[0:-1] for x in lines]
        nodes = np.array([[int(y) for y in x] for x in nodes], dtype=np.int)
        open_file.close()
        # Match edge_indices not wiht ids but with edge_indices in nodes
        node_map = np.zeros(np.max(nodes[:, 0]) + 1, dtype=np.int)
        idx_new = np.arange(len(nodes))
        node_map[nodes[:, 0]] = idx_new
        indexlist = node_map[ids]
        order1 = np.argsort(indexlist[:, 1], axis=0, kind='mergesort')  # stable!
        ind1 = indexlist[order1]
        order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')
        indices = ind1[order2]
        # Class mappings
        class_label_mapping = {'Genetic_Algorithms': 0,
                               'Reinforcement_Learning': 1,
                               'Theory': 2,
                               'Rule_Learning': 3,
                               'Case_Based': 4,
                               'Probabilistic_Methods': 5,
                               'Neural_Networks': 6}
        label_id = np.array([class_label_mapping[x] for x in labels], dtype=np.int)

        self.nodes = nodes
        self.edge_indices = indices
        self.labels_node = label_id
        self.class_label_mapping = class_label_mapping

        return self.nodes, self.edge_indices, self.labels_node, self.class_label_mapping

    def get_graph(self):
        """Make graph tensor objects for Cora dataset after Lu et al. 2003.

        Returns:
            tuple: nodes, indices, labels, class_label_mapping
        """
        return self.nodes, self.edge_indices, self.labels_node, self.class_label_mapping
