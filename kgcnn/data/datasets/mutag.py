import os
import numpy as np

from kgcnn.data.base import GraphDatasetBase


class MUTAGDataset(GraphDatasetBase):

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "MUTAG"
    download_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip"
    file_name = "MUTAG.zip"
    unpack_zip = True
    unpack_directory = "MUTAG"

    def read_in_memory(self):
        path = os.path.join(self.data_main_dir,self.data_directory,self.unpack_directory)
        # adj_matrix
        mutag_a = []
        open_file = open(os.path.join(path, "MUTAG", "MUTAG_A.txt"), "r")
        for lines in open_file.readlines():
            idxs = lines.strip().split(',')
            idxs = [int(x) for x in idxs]
            mutag_a.append(idxs)
        open_file.close()
        mutag_a = np.array(mutag_a)
        # edge_labels
        mutag_e = []
        open_file = open(os.path.join(path, "MUTAG", "MUTAG_edge_labels.txt"), "r")
        for lines in open_file.readlines():
            idxs = int(lines.strip())
            mutag_e.append(idxs)
        open_file.close()
        # graph indicator
        mutag_gi = []
        open_file = open(os.path.join(path, "MUTAG", "MUTAG_graph_indicator.txt"), "r")
        for lines in open_file.readlines():
            idxs = int(lines.strip())
            mutag_gi.append(idxs)
        open_file.close()
        # graph labels
        mutag_gl = []
        open_file = open(os.path.join(path, "MUTAG", "MUTAG_graph_labels.txt"), "r")
        for lines in open_file.readlines():
            idxs = int(lines.strip())
            mutag_gl.append(idxs)
        open_file.close()
        # node labels
        mutag_n = []
        open_file = open(os.path.join(path, "MUTAG", "MUTAG_node_labels.txt"), "r")
        for lines in open_file.readlines():
            idxs = int(lines.strip())
            mutag_n.append(idxs)
        open_file.close()

        # cast to numpy
        mutag_a = np.array(mutag_a, dtype=np.int)
        mutag_e = np.array(mutag_e, dtype=np.int)
        mutag_gi = np.array(mutag_gi, dtype=np.int)
        mutag_gl = np.array(mutag_gl, dtype=np.int)
        mutag_n = np.array(mutag_n, dtype=np.int)

        # labels
        labels = np.array(mutag_gl, dtype=np.int)
        n_data = len(labels)

        # shift index
        mutag_a = mutag_a - 1
        mutag_gi = mutag_gi - 1

        # split into seperate graphs
        graph_id, counts = np.unique(mutag_gi, return_counts=True)
        graphlen = np.zeros(n_data, dtype=np.int)
        graphlen[graph_id] = counts
        nodes0123 = np.split(mutag_n, np.cumsum(graphlen)[:-1])
        node_translate = np.array([6, 7, 8, 9, 53, 17, 35], dtype=np.int)
        # atoms_translate = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
        nodes = [node_translate[x] for x in nodes0123]
        # atoms = [[atoms_translate[y] for y in x] for x in nodes0123]

        # edge_indicator
        graph_id_edge = mutag_gi[mutag_a[:, 0]]  # is the same for adj_matrix[:,1]
        graph_id2, counts_edge = np.unique(graph_id_edge, return_counts=True)
        edgelen = np.zeros(n_data, dtype=np.int)
        edgelen[graph_id2] = counts_edge
        edges = np.split(mutag_e, np.cumsum(edgelen)[:-1])

        # edge_indices
        node_index = np.concatenate([np.arange(x) for x in graphlen], axis=0)
        edge_indices = node_index[mutag_a]
        edge_indices = np.split(edge_indices, np.cumsum(edgelen)[:-1])

        # Check if unconnected
        all_cons = []
        for i in range(len(nodes)):
            cons = np.arange(len(nodes[i]))
            test_cons = np.sort(np.unique(cons[edge_indices[i]].flatten()))
            is_cons = np.zeros_like(cons, dtype=np.bool)
            is_cons[test_cons] = True
            all_cons.append(np.sum(is_cons == False))
        all_cons = np.array(all_cons)
        print("Datainfo: Mol index which has unconnected", np.arange(len(all_cons))[all_cons > 0], "with",
              all_cons[all_cons > 0], "in total", len(all_cons[all_cons > 0]))

        # Set Graph props
        self.labels = labels
        self.nodes = nodes
        self.edge_indices = edge_indices
        self.edges = edges

        return self.labels, self.nodes, self.edge_indices, self.edges


    def get_graph(self):
        return self.labels, self.nodes, self.edge_indices, self.edges