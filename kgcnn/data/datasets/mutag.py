import numpy as np

from kgcnn.data.tudataset import GraphTUDataset


class MUTAGDataset(GraphTUDataset):
    """Store and process MUTAG dataset."""

    def __init__(self, reload=False, verbose=1):
        r"""Initialize MUTAG dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        super(MUTAGDataset, self).__init__("MUTAG", reload=reload, verbose=verbose)

    def read_in_memory(self, verbose=1):
        r"""Load MUTAG data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(MUTAGDataset, self).read_in_memory(verbose=verbose)

        # split into separate graphs
        # graph_id, counts = np.unique(mutag_gi, return_counts=True)
        # graphlen = np.zeros(n_data, dtype=np.int)
        # graphlen[graph_id] = counts
        # nodes0123 = np.split(mutag_n, np.cumsum(graphlen)[:-1])
        node_translate = np.array([6, 7, 8, 9, 53, 17, 35], dtype=np.int)
        atoms_translate = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
        self.node_attributes = [node_translate[np.array(x, dtype="int")][:, 0] for x in self.node_labels]
        # nodes = [node_translate[x] for x in nodes0123]
        atoms = [[atoms_translate[int(y)] for y in x] for x in self.node_labels]

        self.edge_attributes = [x[:, 0] for x in self.edge_labels]
        self.node_symbol = atoms
        self.node_number = self.node_attributes
        self.graph_labels[self.graph_labels < 0] = 0
        self.graph_attributes = None  # We make a better graph attributes here
        self.graph_size = [len(x) for x in self.node_attributes]

        return self
# data = MUTAGDataset()