import numpy as np
import pymatgen
import pymatgen.io.cif
from pymatgen.core.structure import Structure
from kgcnn.crystal.base import CrystalPreprocessor
from . import graph_builder
from networkx import MultiDiGraph


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

    def __init__(self, radius=3.0, size=[3, 3, 3]):
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

    def __init__(self, k=12, size=[3, 3, 3]):
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

    def __init__(self, size=[3, 3, 3]):
        self.size = size

    def __call__(self, structure: Structure) -> MultiDiGraph:
        g = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
        g = graph_builder.add_voronoi_bonds(g, inplace=True)
        g = graph_builder.add_edge_information(g, inplace=True)
        g = graph_builder.to_supercell_graph(g, size=self.size)
        return g


def parse_cif_file_to_structures(cif_file: str):
    # structure = pymatgen.io.cif.CifParser.from_string(cif_string).get_structures()[0]
    structures = pymatgen.io.cif.CifParser(cif_file).get_structures()
    return structures
