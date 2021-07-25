import numpy as np
from kgcnn.mol.methods import coordinates_to_distancematrix, define_adjacency_from_distance, invert_distance, distance_to_gaussdistance

class GeometricMolGraph:
    """Geometric graph of a molecule, e.g. no chemical information just geometric definition."""

    atomic_number = {}

    def __init__(self, atom_labels=None, coordinates=None, atom_number=None):

        self.atom_labels = atom_labels
        self.coordinates = coordinates
        self.atom_number = atom_number
        self.edge_indices = None
        self.angle_indices = None

        if self.atom_labels is not None:
            self.atom_number = [x for x in self.atom_labels]

    def mol_from_xyz(self, xyz: str):
        pass

    def define_graph(self, max_distance, max_neighbours=np.inf, exclusive=True, self_loops=False, do_invert_distance=False,
                     gauss_distance=None):

        if self.coordinates is None:
            print("WARNING:KGCNN: Coordinates are not set for `GeometricMolGraph`. Can not make graph.")
            return None, None

        xyz = self.coordinates
        dist = coordinates_to_distancematrix(xyz)

        # cons = get_connectivity_from_inversedistancematrix(invdist,ats)
        cons, indices = define_adjacency_from_distance(dist, max_distance=max_distance, max_neighbours=max_neighbours, exclusive=exclusive, self_loops=self_loops)
        mask = np.array(cons, dtype=np.bool)
        dist_masked = dist[mask]

        if do_invert_distance:
            dist_masked = invert_distance(dist_masked)
        if gauss_distance is not None:
            dist_masked = distance_to_gaussdistance(dist_masked, gbins=gauss_distance['gbins'],
                                                    grange=gauss_distance['grange'],
                                                    gsigma=gauss_distance['gsigma'])
        # Need at least on feature dimension
        if len(dist_masked.shape) <= 1:
            dist_masked = np.expand_dims(dist_masked, axis=-1)
        edges = dist_masked

        return indices, edges

    def define_angle_indices(self):
        pass

    def to_networkx_graph(self):
        pass

    def to_tensor(self):
        pass
