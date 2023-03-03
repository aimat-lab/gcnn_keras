import os
import numpy as np

from kgcnn.data.datasets.GraphTUDataset2020 import GraphTUDataset2020


class MutagenicityDataset(GraphTUDataset2020):
    r"""Store and process Mutagenicity dataset from `TUDatasets <https://chrsmrrs.github.io/datasets/>`__ .

    Mutagenicity is a chemical compound dataset of drugs, which can be categorized into two classes:
    mutagen and non-mutagen.

    References:

        (1) Riesen, K. and Bunke, H.: IAM Graph Database Repository for Graph Based Pattern Recognition and
            Machine Learning. In: da Vitora Lobo, N. et al. (Eds.), SSPR&SPR 2008, LNCS, vol. 5342, pp. 287-297, 2008.

    """

    def __init__(self, reload=False, verbose: int = 10):
        """Initialize Mutagenicity dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        # Use default base class init()
        super(MutagenicityDataset, self).__init__("Mutagenicity", reload=reload, verbose=verbose)

    def read_in_memory(self):
        r"""Load Mutagenicity Dataset into memory and already split into items with further cleaning and
        processing.
        """
        super(MutagenicityDataset, self).read_in_memory()

        node_translate = np.array([6, 8, 17, 1, 7, 9, 35, 16, 15, 53, 11, 19, 3, 20], dtype="int")
        atoms_translate = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'ksb', 'Li', 'Ca']
        z_translate = {node_translate[i]: atoms_translate[i] for i in range(len(node_translate))}

        edge_indices = self.obtain_property("edge_indices")
        node_labels = self.obtain_property("node_labels")
        edge_labels = self.obtain_property("edge_labels")
        graph_labels = self.obtain_property("graph_labels")

        nodes = [node_translate[np.array(x, dtype="int")][:, 0] for x in node_labels]
        atoms = [[atoms_translate[int(y[0])] for y in x] for x in node_labels]
        edges = [x[:, 0]+1 for x in edge_labels]
        labels = graph_labels

        # Require cleaning steps
        labels_clean = []
        nodes_clean = []
        edge_indices_clean = []
        edges_clean = []
        atoms_clean = []

        # Remove unconnected atoms. not Na Li etc.
        self.info("Checking database...")
        for i in range(len(nodes)):
            nats = nodes[i]
            cons = np.arange(len(nodes[i]))
            test_cons = np.sort(np.unique(edge_indices[i].flatten()))
            is_cons = np.zeros_like(cons, dtype="bool")
            is_cons[test_cons] = True
            is_cons[nats == 20] = True  # Allow to be unconnected
            is_cons[nats == 3] = True  # Allow to be unconnected
            is_cons[nats == 19] = True  # Allow to be unconnected
            is_cons[nats == 11] = True  # Allow to be unconnected
            if np.sum(is_cons) != len(cons):
                info_list = nodes[i][is_cons == False]
                info_list, info_cnt = np.unique(info_list, return_counts=True)
                info_list = {z_translate[info_list[j]]: info_cnt[j] for j in range(len(info_list))}
                self.info("Removing unconnected %s from molecule %s" % (info_list, i))
                nodes_clean.append(nats[is_cons])
                atoms_clean.append([atoms[i][j] for j in range(len(is_cons)) if is_cons[j] == True])
                # Need to correct edge_indices
                indices_used = cons[is_cons]
                indices_new = np.arange(len(indices_used))
                indices_old = np.zeros(len(nodes[i]), dtype="int")
                indices_old[indices_used] = indices_new
                edge_idx_new = indices_old[edge_indices[i]]
                edge_indices_clean.append(edge_idx_new)
            else:
                nodes_clean.append(nats)
                atoms_clean.append(atoms[i])
                edge_indices_clean.append(edge_indices[i])
            edges_clean.append(edges[i])
            labels_clean.append(labels[i])

        self.info("Database still has unconnected Na+, Li+, ksb+ etc.")

        # Since no attributes in graph dataset, we use labels as attributes
        self.assign_property("graph_labels", labels_clean)
        self.assign_property("edge_indices", edge_indices_clean)
        self.assign_property("node_attributes", nodes_clean)
        self.assign_property("edge_attributes", edges_clean)
        self.assign_property("node_labels", nodes_clean)
        self.assign_property("edge_labels", edges_clean)
        self.assign_property("node_symbol", atoms_clean)
        self.assign_property("node_number", nodes_clean)
        self.assign_property("graph_size", [len(x) for x in nodes_clean])

        # return labels,nodes,edge_indices,edges,atoms
        return self
