import numpy as np
try:
    import pymatgen
    import pymatgen.io.cif
except ImportError:
    print("This module needs pymatgen to be installed.")

from kgcnn.utils.adj import sort_edge_indices


def parse_cif_file_to_structures(cif_file: str):
    # structure = pymatgen.io.cif.CifParser.from_string(cif_string).get_structures()[0]
    structures = pymatgen.io.cif.CifParser(cif_file).get_structures()
    return structures


def structure_get_properties(py_struct):
    node_coordinates = np.array(py_struct.cart_coords, dtype="float")
    lattice_matrix = np.ascontiguousarray(np.array(py_struct.lattice.matrix), dtype="float")
    abc = np.array(py_struct.lattice.abc)
    charge = np.array([py_struct.charge], dtype="float")
    volume = np.array([py_struct.lattice.volume], dtype="float")
    occupation = np.zeros((len(py_struct.sites), 95))
    oxidation = np.zeros((len(py_struct.sites), 95))
    for i, x in enumerate(py_struct.sites):
        for sp, occ in x.species.items():
            occupation[i, sp.number] = occ
            oxidation[i, sp.number] = sp.oxi_state
    symbols = np.array([x.species_string for x in py_struct.sites])

    return [node_coordinates, lattice_matrix, abc, charge, volume, occupation, oxidation, symbols]


def structure_get_range_neighbors(py_struct, radius=4,  numerical_tol: float = 1e-08, struct_id=None,
                                  max_neighbours: int = 100000000):
    # Determine all neighbours
    all_nbrs = py_struct.get_all_neighbors(radius, include_index=True, numerical_tol=numerical_tol)

    all_edge_distance = []
    all_edge_indices = []
    all_edge_image = []
    for i, start_site in enumerate(all_nbrs):
        edge_distance = []
        edge_indices = []
        edge_image = []
        for j, stop_site in enumerate(start_site):
            edge_distance.append(stop_site.nn_distance)
            edge_indices.append([i, stop_site.index])
            edge_image.append(stop_site.image)
        # Sort after distance
        edge_distance = np.array(edge_distance)
        order_dist = np.argsort(edge_distance)
        edge_distance = np.expand_dims(edge_distance[order_dist], axis=-1)
        edge_indices = np.array(edge_indices, dtype="int")[order_dist]
        edge_image = np.array(edge_image, dtype="int")[order_dist]
        # Append to index list
        all_edge_distance.append(edge_distance[:max_neighbours])
        all_edge_indices.append(edge_indices[:max_neighbours])
        all_edge_image.append(edge_image[:max_neighbours])

    all_edge_distance = np.concatenate(all_edge_distance, axis=0)
    all_edge_indices = np.concatenate(all_edge_indices, axis=0)
    all_edge_image = np.concatenate(all_edge_image, axis=0)
    # Sort after edge indices again.
    edge_indices, edge_image, edge_distance = sort_edge_indices(all_edge_indices, all_edge_image, all_edge_distance)
    return [edge_indices, edge_image, edge_distance]
