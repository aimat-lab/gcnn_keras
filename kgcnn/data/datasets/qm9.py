import os
import pickle
import numpy as np
import shutil

from kgcnn.data.mol.methods import coordinates_to_distancematrix, invert_distance, distance_to_gaussdistance, \
    define_adjacency_from_distance, get_angle_indices

from kgcnn.data.base import GraphDatasetBase

class QM9Dataset(GraphDatasetBase):

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "qm9"
    download_url = "https://ndownloader.figshare.com/files/3195389"
    file_name = 'dsgdb9nsd.xyz.tar.bz2'
    unpack_tar = True
    unpack_zip = False
    unpack_directory = 'dsgdb9nsd.xyz'
    fits_in_memory = True
    process_dataset = True


    def prepare_data(self, overwrite=False):
        path = os.path.join(self.data_main_dir, self.data_directory)

        datasetsize = 133885
        qm9 = []

        if os.path.exists(os.path.join(path, "qm9.pickle")) and not overwrite:
            print("INFO: Single molecules already pickled... done")
            return qm9

        if not os.path.exists(os.path.join(path, 'dsgdb9nsd.xyz')):
            print("ERROR: Can not find extracted dsgdb9nsd.xyz directory")
            return qm9

        print("INFO: Reading dsgdb9nsd files ...", end='', flush=True)
        for i in range(1, datasetsize + 1):
            mol = []
            file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
            open_file = open(os.path.join(path, "dsgdb9nsd.xyz", file), "r")
            lines = open_file.readlines()
            mol.append(int(lines[0]))
            labels = lines[1].strip().split(' ')[1].split('\t')
            if int(labels[0]) != i:
                print("Warning index not matching xyz-file.")
            labels = [int(labels[0])] + [float(x) for x in labels[1:]]
            mol.append(labels)
            cords = []
            for j in range(int(lines[0])):
                atom_info = lines[2 + j].strip().split('\t')
                cords.append([atom_info[0]] + [float(x.replace('*^', 'e')) for x in atom_info[1:]])
            mol.append(cords)
            freqs = lines[int(lines[0]) + 2].strip().split('\t')
            freqs = [float(x) for x in freqs]
            mol.append(freqs)
            smiles = lines[int(lines[0]) + 3].strip().split('\t')
            mol.append(smiles)
            inchis = lines[int(lines[0]) + 4].strip().split('\t')
            mol.append(inchis)
            open_file.close()
            qm9.append(mol)
            # Remove file after reading

        # save
        print('done')
        print("INFO: Saving qm9.pickle ...", end='', flush=True)
        with open(os.path.join(path, "qm9.pickle"), 'wb') as f:
            pickle.dump(qm9, f)
        print('done')

        print("INFO: Cleaning up extracted files...", end='', flush=True)
        for i in range(1, datasetsize + 1):
            file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
            file = os.path.join(path, "dsgdb9nsd.xyz", file)
            os.remove(file)
        print('done')

        return qm9

    def read_in_memory(self):
        path = os.path.join(self.data_main_dir, self.data_directory)

        print("INFO: Reading dataset ...", end='', flush=True)
        with open(os.path.join(path, "qm9.pickle"), 'rb') as f:
            qm9 = pickle.load(f)

        # labels
        labels = np.array([x[1] for x in qm9])

        # Atoms as nodes
        atoms = [[y[0] for y in x[2]] for x in qm9]
        # nodelens = np.array([len(x) for x in atoms], dtype=np.int)
        atom_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        # atom_1hot = {'H': [1, 0, 0, 0, 0], 'C': [0, 1, 0, 0, 0], 'N': [0, 0, 1, 0, 0], 'O': [0, 0, 0, 1, 0],
        #              'F': [0, 0, 0, 0, 1]}
        zval = [[atom_dict[y] for y in x] for x in atoms]
        outzval = [np.array(x, dtype=np.int) for x in zval]
        # outatoms = np.concatenate(outatom,axis=0)
        # a1hot = [[atom_1hot[y] for y in x] for x in atoms]
        # outa1hot = [np.array(x, dtype=np.float32) for x in a1hot]
        nodes = outzval

        # States
        massdict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984}
        mass = [[massdict[y] for y in x] for x in atoms]
        mmw = np.expand_dims(np.array([np.mean(x) for x in mass]), axis=-1)

        # Edges
        coord = [[[y[1], y[2], y[3]] for y in x[2]] for x in qm9]
        coord = [np.array(x) for x in coord]

        self.coord = coord
        self.labels = labels
        self.atoms = atoms
        self.nodes = nodes
        self.mmw = mmw

        print('done')

    def get_graph(self,max_distance=4, max_neighbours=15,
                  do_invert_distance= False, do_gauss_basis_expansion= True,
                  gauss_distance=None, max_mols=133885):
        """Make graph objects from qm9 dataset.

        Args:
            max_distance (int): 4
            max_neighbours (int): 15
            gauss_distance (dict): None
            max_mols (int): Maximum number of molecules to take from qm9. Default is 133885.

        Returns:
            list: [labels, nodes, edges, edge_idx, gstates]

            - labels: All labels of qm9
            - nodes: List of atomic numbers for emebdding layer
            - edges: Edgefeatures (inverse distance, gauss distance)
            - edge_idx: Edge indices (N,2)
            - gstates: Graph states, mean moleculare weight - 7 g/mol
        """

        if gauss_distance is None:
            gauss_distance = {'gbins': 20, 'grange': 4, 'gsigma': 0.4}

        coord = self.coord
        labels = self.labels
        nodes = self.nodes
        gstates = self.mmw
        gstates = gstates - 7.0  # center at 0

        edge_idx = []
        edges = []

        for i in range(max_mols):
            xyz = coord[i]
            dist = coordinates_to_distancematrix(xyz)

            # cons = get_connectivity_from_inversedistancematrix(invdist,ats)
            cons, _ = define_adjacency_from_distance(dist, max_distance=max_distance, max_neighbours=max_neighbours,
                                                     exclusive=True, self_loops=False)
            index1 = np.tile(np.expand_dims(np.arange(0, dist.shape[0]), axis=1), (1, dist.shape[1]))
            index2 = np.tile(np.expand_dims(np.arange(0, dist.shape[1]), axis=0), (dist.shape[0], 1))
            mask = np.array(cons, dtype=np.bool)
            index12 = np.concatenate([np.expand_dims(index1, axis=-1), np.expand_dims(index2, axis=-1)], axis=-1)
            edge_idx.append(index12[mask])
            dist_masked = dist[mask]

            if do_invert_distance:
                dist_masked = invert_distance(dist_masked)
            if do_gauss_basis_expansion:
                dist_masked = distance_to_gaussdistance(dist_masked, gbins=gauss_distance['gbins'],
                                                        grange=gauss_distance['grange'],
                                                        gsigma=gauss_distance['gsigma'])
            # Need at least on feature dimension
            if len(dist_masked.shape) <= 1:
                dist_masked = np.expand_dims(dist_masked, axis=-1)

            edges.append(dist_masked)

        # edge_len = np.array([len(x) for x in edge_idx], dtype=np.int)
        # edges = [np.concatenate([edges_inv[i],edges[i]],axis=-1) for i in range(len(edge_idx))]
        # edges = [edges[i] for i in range(len(edge_idx))]
        # self.edge_index = edge_idx[:max_mols]

        return labels[:max_mols], nodes[:max_mols], edges[:max_mols], edge_idx[:max_mols], gstates[:max_mols]

    @classmethod
    def get_angle_index(cls, idx, is_sorted = False):
        ei = []
        nijk = []
        ai = []
        for x in idx:
            temp = get_angle_indices(x,is_sorted=is_sorted)
            ei.append(temp[0])
            nijk.append(temp[1])
            ai.append(temp[2])

        return ei, nijk, ai
