from pymatgen.core.structure import Structure
from typing import Callable
from hashlib import md5
from networkx import MultiDiGraph
from . import graph_builder

class CrystalPreprocessor(Callable[[Structure], MultiDiGraph]):

    def __call__(self, structure: Structure) -> MultiDiGraph:
        raise NotImplementedError()

    def get_config(self):
        config = vars(self)
        config = {k:v for k,v in config.items() if not k.startswith("_")}
        config['preprocessor'] = self.__class__.__name__
        return config

    def hash(self):
        return md5(str(self.get_config()).encode()).hexdigest()

    def __hash__(self):
        return int(self.hash(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)

class RadiusAsymmetricUnitCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords', 'multiplicity']
    edge_attributes = ['cell_translation', 'distance', 'symmop', 'offset']
    graph_attributes = ['lattice_matrix', 'spacegroup']

    def __init__(self, radius=3.0):
        self.radius = radius

    def __call__(self, structure: Structure):
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=True)
        g = graph_builder.add_radius_bonds(g, radius=self.radius, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_asymmetric_unit_graph(g)
        return g

class KNNAsymmetricUnitCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords', 'multiplicity']
    edge_attributes = ['cell_translation', 'distance', 'symmop', 'offset']
    graph_attributes = ['lattice_matrix', 'spacegroup']

    def __init__(self, k=12):
        self.k = k

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=True)
        g = graph_builder.add_knn_bonds(g, k=self.k, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_asymmetric_unit_graph(g)
        return g

class VoronoiAsymmetricUnitCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords', 'multiplicity']
    edge_attributes = ['cell_translation', 'distance', 'symmop', 'offset']
    graph_attributes = ['lattice_matrix', 'spacegroup']

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=True)
        g = graph_builder.add_voronoi_bonds(g, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_asymmetric_unit_graph(g)
        return g


class RadiusUnitCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, radius=3.0):
        self.radius = radius

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_radius_bonds(g, radius=self.radius, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g


class KNNUnitCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, k=12):
        self.k = k

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_knn_bonds(g, k=self.k, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g


class VoronoiUnitCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_voronoi_bonds(g, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        return g
        

class RadiusSuperCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, radius=3.0, size=[3,3,3]):
        self.radius = radius
        self.size = size

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_radius_bonds(g, radius=self.radius, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_supercell_graph(g, size=self.size)
        return g

class KNNSuperCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, k=12, size=[3,3,3]):
        self.size = size
        self.k = k

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_knn_bonds(g, k=self.k, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_supercell_graph(g, size=self.size)
        return g
        
class VoronoiSuperCell(CrystalPreprocessor):

    node_attributes = ['atomic_number', 'frac_coords', 'coords']
    edge_attributes = ['cell_translation', 'distance', 'offset']
    graph_attributes = ['lattice_matrix']

    def __init__(self, size=[3,3,3]):
        self.size = size

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_voronoi_bonds(g, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_supercell_graph(g, size=self.size)
        return g

