import numpy as np

from kgcnn.data.datasets.GraphTUDataset2020 import GraphTUDataset2020


class MUTAGDataset(GraphTUDataset2020):
    """Store and process MUTAG dataset."""

    def __init__(self, reload=False, verbose=1):
        r"""Initialize MUTAG dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        super(MUTAGDataset, self).__init__("MUTAG", reload=reload, verbose=verbose)

    def read_in_memory(self, verbose: int = 1):
        r"""Load MUTAG data into memory and already split into items."""
        super(MUTAGDataset, self).read_in_memory()

        edge_labels = self.obtain_property("edge_labels")
        node_labels = self.obtain_property("node_labels")
        graph_labels = self.obtain_property("graph_labels")
        # split into separate graphs
        # graph_id, counts = np.unique(mutag_gi, return_counts=True)
        # graphlen = np.zeros(n_data, dtype=np.int)
        # graphlen[graph_id] = counts
        # nodes0123 = np.split(mutag_n, np.cumsum(graphlen)[:-1])
        node_translate = np.array([6, 7, 8, 9, 53, 17, 35], dtype="int")
        atoms_translate = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
        node_attributes = [node_translate[np.array(x, dtype="int")][:, 0] for x in node_labels]
        # nodes = [node_translate[x] for x in nodes0123]
        atoms = [[atoms_translate[int(y)] for y in x] for x in node_labels]
        graph_labels = np.array(graph_labels)
        graph_labels[graph_labels < 0] = 0

        self.assign_property("node_attributes", node_attributes)
        self.assign_property("edge_attributes", [x[:, 0] for x in edge_labels])
        self.assign_property("node_symbol", atoms)
        self.assign_property("node_number", node_attributes)
        self.assign_property("graph_labels", [x for x in graph_labels])
        self.assign_property("graph_size", [len(x) for x in node_attributes])

        return self

# data = MUTAGDataset()
