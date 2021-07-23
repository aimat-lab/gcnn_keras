import numpy as np


class GeometricMolGraph:
    """Geometric graph of a molecule, e.g. no chemical information just geometric definition."""

    def __init__(self, atom_labels=None, coordinates=None):

        self.atom_labels = atom_labels
        self.coordinates = coordinates
        self.atom_number = None
        self.edge_indices = None

        if self.atom_labels is not None:
            self.atom_number = [x for x in self.atom_labels]

    def mol_from_xyz(self, xyz: str):
        pass

    def define_graph(self):
        pass

    def to_networkx_graph(self):
        pass

    def to_tensor(self):
        pass
