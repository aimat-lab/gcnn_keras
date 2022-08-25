import pyxtal
# pyxtal requires pymatgen as dependency.
from pymatgen.core.structure import Structure
from networkx import MultiDiGraph
from kgcnn.crystal.base import CrystalPreprocessor
from . import graph_builder


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