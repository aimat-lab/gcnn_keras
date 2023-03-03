import numpy as np

from kgcnn.data.datasets.GraphTUDataset2020 import GraphTUDataset2020


class MUTAGDataset(GraphTUDataset2020):
    r"""Store and process MUTAG dataset from `TUDatasets <https://chrsmrrs.github.io/datasets/>`__ .

    In `Papers with Code <https://paperswithcode.com/dataset/mutag>`__ :
    In particular, MUTAG is a collection of nitroaromatic compounds and the goal is to predict their mutagenicity
    on Salmonella typhimurium. Input graphs are used to represent chemical compounds, where vertices stand for atoms
    and are labeled by the atom type (represented by one-hot encoding), while edges between vertices represent
    bonds between the corresponding atoms. It includes 188 samples of chemical compounds with 7 discrete node labels.

    References:

        (1) Debnath, A.K., Lopez de Compadre, R.L., Debnath, G., Shusterman, A.J., and Hansch, C.
            Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds.
            J. Med. Chem. 34(2):786-797 (1991).
        (2) Nils Kriege, Petra Mutzel. 2012. Subgraph Matching Kernels for Attributed Graphs.
            International Conference on Machine Learning 2012.

    """

    def __init__(self, reload=False, verbose: int = 10):
        r"""Initialize MUTAG dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
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
