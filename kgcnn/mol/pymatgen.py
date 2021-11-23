import pymatgen
import pymatgen.io.cif
import numpy as np
from kgcnn.utils.adj import sort_edge_indices


def parse_cif_file_to_structures(cif_file: str):
    # structure = pymatgen.io.cif.CifParser.from_string(cif_string).get_structures()[0]
    structures = pymatgen.io.cif.CifParser(cif_file).get_structures()
    return structures


def convert_structures_as_dict(structures: list):
    dicts = [s.as_dict() for s in structures]
    return dicts


def structure_get_properties(py_struct):
    node_coordinates = np.array(py_struct.cart_coords, dtype="float")
    lattice_matrix = np.ascontiguousarray(np.array(py_struct.lattice.matrix), dtype="float")
    abc = np.array(py_struct.lattice.abc)
    charge = np.array([py_struct.charge], dtype="float")
    volume = np.array([py_struct.lattice.volume], dtype="float")
    symbols = np.array([x.species_string for x in py_struct.sites])

    return [node_coordinates, lattice_matrix, abc, charge, volume, symbols]


def structure_get_range_neighbors(py_struct, radius=4,  numerical_tol: float = 1e-08, struct_id=None,
                                  max_neighbours: int = 100000000):
    # Determine all neighbours
    all_nbrs = py_struct.get_all_neighbors(radius, include_index=True, numerical_tol=numerical_tol)

    edge_distance = []
    edge_indices = []
    edge_image = []
    for i, start_site in enumerate(all_nbrs):
        for j, stop_site in enumerate(start_site):
            edge_distance.append(stop_site.nn_distance)
            edge_indices.append([i, stop_site.index])
            edge_image.append(stop_site.image)

    # Sort after distance
    edge_distance = np.array(edge_distance)
    order_distance = np.argsort(edge_distance)
    # Sort image and indices too
    edge_distance = edge_distance[order_distance]
    edge_indices = np.array(edge_indices, dtype="int")[order_distance]
    edge_image = np.array(edge_image, dtype="int")[order_distance]
    # Pick max neighbours
    edge_distance = edge_distance[:max_neighbours]
    edge_indices = edge_indices[:max_neighbours]
    edge_image = edge_image[:max_neighbours]
    edge_distance = np.expand_dims(np.array(edge_distance), axis=-1)

    if len(edge_indices) <= 0:
        print("ERROR:kgcnn: No edges for %s, please increase cutoff or neighbours" % struct_id)
        return [edge_indices, edge_image, edge_distance]

    # Sort after edge indices.
    edge_indices, edge_image, edge_distance = sort_edge_indices(edge_indices, edge_image, edge_distance)
    return [edge_indices, edge_image, edge_distance]
