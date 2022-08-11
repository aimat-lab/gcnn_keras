import warnings
from copy import deepcopy
import numpy as np
from scipy.spatial import Voronoi
from networkx import MultiDiGraph
from pyxtal import pyxtal
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.optimization.neighbors import find_points_in_spheres
from typing import Union

def get_symmetrized_graph(structure: Union[Structure, pyxtal]) -> MultiDiGraph:

    graph = MultiDiGraph()
    if isinstance(structure, pyxtal):
        pyxtal_cell = structure 
    elif isinstance(structure, Structure):
        try:
            pyxtal_cell = pyxtal()
            pyxtal_cell.from_seed(structure)
        except:
            # use trivial spacegroup (with spacegroup number == 1)
            # if spglib isn't able to calculate symmetries
            frac_coords = np.array([site.frac_coords for site in structure.sites])
            frac_coords = _to_unit_cell(frac_coords)
            for node_idx, site in enumerate(structure.sites):
                graph.add_node(node_idx, atomic_number=site.specie.number,
                        asymmetric_mapping=node_idx,
                        frac_coords=frac_coords[node_idx],
                        coords=site.coords,
                        symmop=np.eye(4),
                        multiplicity=1)
            setattr(graph, 'lattice_matrix', structure.lattice.matrix)
            setattr(graph, 'spacegroup', 1)
            return graph
    else:
        ValueError("This method takes either a pymatgen.core.structure.Structure or a pyxtal object.")

    atomic_numbers, frac_coords, asymmetric_mapping, symmops, multiplicities = [], [], [], [], []
    for site in pyxtal_cell.atom_sites:
        atomic_numbers += (site.multiplicity * [Element(site.specie).Z])
        asymmetric_mapping += (site.multiplicity * [len(asymmetric_mapping)])
        frac_coords.append(site.coords)
        symmops += [symmop.affine_matrix for symmop in site.wp.ops]
        multiplicities += (site.multiplicity * [site.multiplicity])
    frac_coords = _to_unit_cell(np.vstack(frac_coords))
    lattice = pyxtal_cell.lattice.matrix
    coords = frac_coords @ lattice
    for node_idx in range(len(atomic_numbers)):
        graph.add_node(node_idx, atomic_number=atomic_numbers[node_idx],
                asymmetric_mapping=asymmetric_mapping[node_idx],
                frac_coords=frac_coords[node_idx],
                coords=coords[node_idx],
                symmop= symmops[node_idx],
                multiplicity=multiplicities[node_idx])
    setattr(graph, 'lattice_matrix', lattice)
    setattr(graph, 'spacegroup', pyxtal_cell.group.number)
    
    return graph

def structure_to_empty_graph(structure: Union[Structure, pyxtal], symmetrize=False) -> MultiDiGraph:

    if symmetrize:
        return get_symmetrized_graph(structure)
    else:
        if isinstance(structure, pyxtal):
            structure = structure.to_pymatgen()
        graph = MultiDiGraph()
        frac_coords = np.array([site.frac_coords for site in structure.sites])
        frac_coords = _to_unit_cell(frac_coords)
        for node_idx, site in enumerate(structure.sites):
            graph.add_node(node_idx, atomic_number=site.specie.number,
                    frac_coords=frac_coords[node_idx],
                    coords=site.coords)
        setattr(graph, 'lattice_matrix', structure.lattice.matrix)
        return graph

def add_knn_bonds(graph: MultiDiGraph, k=12, max_radius=10., inplace=False):

    lattice = deepcopy(graph.lattice_matrix)
    frac_coords = np.array([data[1] for data in graph.nodes(data='frac_coords')])
    coords = frac_coords @ lattice
    index1, index2, offset_vectors, distances = find_points_in_spheres(coords,
                                                                   coords,
                                                                   r=max_radius,
                                                                   pbc=np.array([True] * 3, dtype=int),
                                                                   lattice=lattice,
                                                                   tol=1e-8)
    # Remove self_loops:
    no_self_loops = np.argwhere(~np.isclose(distances, 0)).reshape(-1)
    index1 = index1[no_self_loops]
    index2 = index2[no_self_loops]
    offset_vectors = offset_vectors[no_self_loops]
    distances = distances[no_self_loops]
    
    new_graph = graph if inplace else deepcopy(graph)

    for node_idx in range(new_graph.number_of_nodes()):
        idxs = np.argwhere(index1 == node_idx)[:,0]
        sorted_idxs = idxs[np.argsort(distances[idxs])]
        sorted_idxs = sorted_idxs[:k]
        # If the max_radius doesn't capture 12 neighbors, try again with double the max_radius
        if len(sorted_idxs) < k:
            return add_knn_bonds(new_graph, k=k, max_radius=max_radius*2, inplace=True)
        for edge_idx in sorted_idxs:
            new_graph.add_edge(index2[edge_idx], index1[edge_idx], cell_translation=offset_vectors[edge_idx],\
                    distance=distances[edge_idx])

    return new_graph



def add_radius_bonds(graph: MultiDiGraph, radius=5., inplace=False):
    new_graph = graph if inplace else deepcopy(graph)

    lattice = deepcopy(graph.lattice_matrix)
    frac_coords = np.array([data[1] for data in graph.nodes(data='frac_coords')])
    coords = frac_coords @ lattice
    index1, index2, offset_vectors, distances = find_points_in_spheres(coords,
                                                                   coords,
                                                                   r=radius,
                                                                   pbc=np.array([True] * 3, dtype=int),
                                                                   lattice=lattice,
                                                                   tol=1e-8)
    # Remove self_loops:
    no_self_loops = np.argwhere(~np.isclose(distances, 0)).reshape(-1)
    index1 = index1[no_self_loops]
    index2 = index2[no_self_loops]
    offset_vectors = offset_vectors[no_self_loops]
    distances = distances[no_self_loops]

    if len(index1) == 0:
        warnings.warn('No edges added to the graph, \
consider increasing the radius and check your graph input instance.')

    new_graph = deepcopy(graph)
    for source, target, cell_translation, dist in zip(index2, index1, offset_vectors, distances):
        new_graph.add_edge(source, target, cell_translation=cell_translation, distance=dist)

    return new_graph


def add_voronoi_bonds(graph: MultiDiGraph, inplace=False):
    new_graph = graph if inplace else deepcopy(graph)

    lattice = graph.lattice_matrix
    frac_coords = np.array([data[1] for data in graph.nodes(data='frac_coords')])
    dim = lattice.shape[0]
    assert dim == 3
    size = np.array([1,1,1])
    expanded_frac_coords = _get_super_cell_frac_coords(lattice, frac_coords, size)
    expanded_coords = expanded_frac_coords @ lattice
    flattened_expanded_coords = expanded_coords.reshape(-1, dim)

    voronoi = Voronoi(flattened_expanded_coords)
    ridge_points_unraveled = np.array(np.unravel_index(voronoi.ridge_points, expanded_coords.shape[:-1]))
    # shape: (num_ridges, 2 (source, target), 4 (3 cell_index + 1 atom_index)) 
    ridge_points_unraveled = np.moveaxis(ridge_points_unraveled, [0,1,2], [2,0,1])

    # Filter ridges that have source in the centered unit cell
    source_in_center_cell = np.argwhere(np.all(ridge_points_unraveled[:,0,:dim] == 1, axis=-1))[:,0]
    # Filter ridges that have target in the centered unit cell
    target_in_center_cell = np.argwhere(np.all(ridge_points_unraveled[:,1,:dim] == 1, axis=-1))[:,0]
    
    edge_info = np.vstack(
            [ridge_points_unraveled[source_in_center_cell][:,[1,0]], ridge_points_unraveled[target_in_center_cell]])
    
    cell_translations = (edge_info[:,0,:-1] - size).astype(float)
    edge_indices = edge_info[:,:,-1]

    distances = []
    for i in range(len(edge_indices)):
        d = np.linalg.norm(expanded_coords[tuple(edge_info[i][0])] - expanded_coords[tuple(edge_info[i][1])])
        distances.append(d)
    
    for nodes, cell_translation, dist in zip(edge_indices, cell_translations, distances):
        source, target = nodes[0], nodes[1]
        new_graph.add_edge(source, target, cell_translation=cell_translation, distance=dist)
    
    return new_graph

def remove_duplicate_edges(graph: MultiDiGraph, inplace=False) -> MultiDiGraph:
    new_graph = graph if inplace else deepcopy(graph)
    edge_counter = set()
    remove_edges = set()
    for e in new_graph.edges(data='cell_translation', keys=True):
        id_ = (e[0], e[1]) + tuple(e[3].astype(int))
        if id_ in edge_counter:
            remove_edges.add((e[0], e[1], e[2]))
        else:
            edge_counter.add(id_)
    for edge in remove_edges:
        new_graph.remove_edge(edge[0], edge[1], key=edge[2])
    return new_graph


def get_mesh(size, dim):
    if isinstance(size, int):
        size = [size] * dim
    else:
        size = list(size)
    assert len(size) == dim
    
    mesh = np.array(np.meshgrid(*tuple([np.arange(i) for i in size])))
    mesh = np.moveaxis(mesh, [0,1], [-1,1])
    return mesh

def get_cube(dim: int):
    return get_mesh(2, dim)

def _get_max_diameter(lattice):
    dim = lattice.shape[0]
    cube = get_cube(dim)
    max_radius = np.max(np.linalg.norm((cube - 1/2) @ lattice, axis=1))
    return max_radius * 2

def _get_super_cell_size_from_radius(lattice: np.ndarray, radius: float):
    dim = lattice.shape[0]
    max_diameter = _get_max_diameter(lattice)
    radius_ = radius + max_diameter
    cell_indices = np.ceil(np.sum(np.abs(np.linalg.inv(lattice)), axis=0) * radius_).astype(int)
    cells = get_mesh(cell_indices + 1, dim)
    lattice_point_coords = cells @ lattice
    images = np.argwhere(np.linalg.norm(lattice_point_coords, axis=-1) <= radius_)
    super_cell_size = images.max(axis=0)
    return super_cell_size

def _get_super_cell_frac_coords(lattice, frac_coords, size):
    dim = lattice.shape[0]
    
    if isinstance(size, int):
        size = [size] * dim
    else:
        size = list(size)
    assert len(size) == dim
    
    doubled_size = np.array(size) * 2 + 1
    mesh = get_mesh(doubled_size, dim)
    # frac_coords_expanded.shape == (1,1,1,num_atoms,3) (for dim == 3)
    frac_coords_expanded = np.expand_dims(frac_coords, np.arange(dim).tolist())
    # mesh_expanded.shape == (double_size[0], double_size[1], double_size[2], 1, 3) (for dim == 3)
    mesh_expanded = np.expand_dims(mesh - size, -2)
    expanded_frac_coords = mesh_expanded + frac_coords_expanded
    
    return expanded_frac_coords



def reshape_at_axis(arr, axis, new_shape):
    # move reshape axis to last axis position, because np.reshape reshapes this first
    arr_tmp = np.moveaxis(arr, axis, -1)
    shape = arr_tmp.shape[:-1] + new_shape
    new_positions = np.arange(len(new_shape)) + axis
    old_positions = np.arange(len(new_shape)) + (len(arr.shape) - 1)
    # now call np.reshape and move axis to right position
    return np.moveaxis(arr_tmp.reshape(shape), old_positions, new_positions)

def pairwise_diff(coords1, coords2):
    # Difference calculated at last axis of both inputs
    assert coords1.shape[-1] == coords2.shape[-1]
    coords1_reshaped = coords1.reshape(-1,coords1.shape[-1])
    coords2_reshaped = coords2.reshape(-1,coords2.shape[-1])
    diffs = np.expand_dims(coords2_reshaped, 0) - np.expand_dims(coords1_reshaped, 1)
    return reshape_at_axis(reshape_at_axis(diffs, 1, coords2.shape[:-1]), 0, coords1.shape[:-1])

def add_edge_information(graph: MultiDiGraph, inplace=False,
        frac_offset=False, offset=True, distance=True) -> MultiDiGraph:

    new_graph = graph if inplace else deepcopy(graph)
    if graph.number_of_edges() == 0:
        return new_graph

    add_frac_offset = frac_offset
    add_offset = offset
    add_distance = distance

    frac_coords1 = []
    frac_coords2 = []
    cell_translations = []
    
    # Collect necessary coordinate informations for calculations
    for e in graph.edges(data='cell_translation'):
        frac_coords1.append(graph.nodes[e[0]]['frac_coords'])
        cell_translations.append(e[2])
        frac_coords2.append(graph.nodes[e[1]]['frac_coords'])

    # Do calculations in vectorized form (instead of doing it inside the edge loop)
    frac_coords1 = np.array(frac_coords1)
    frac_coords2 = np.array(frac_coords2)
    cell_translations = np.array(cell_translations)
    frac_offset = frac_coords2 - (frac_coords1 + cell_translations)
    offset = frac_offset @ graph.lattice_matrix
    if add_distance:
        distances = np.linalg.norm(offset, axis=-1)

    # Add calculated informations to edge attributes
    for i, e in enumerate(graph.edges(data=True)):
        if add_frac_offset:
            e[2]['frac_offset'] = frac_offset[i]
        if add_offset:
            e[2]['offset'] = offset[i]
        if add_distance:
            e[2]['distance'] = distances[i]

    return new_graph

def to_supercell_graph(graph: MultiDiGraph, size):

    supercell_graph = MultiDiGraph()
    size_ = list(size) + [graph.number_of_nodes()]
    new_num_nodes = np.prod(size_)
    for node in range(new_num_nodes):
        idx = np.unravel_index(node, size_)
        cell_translation = idx[:3]
        node_num = idx[3]
        data = deepcopy(graph.nodes[node_num])
        data['frac_coords'] = data['frac_coords'] + np.array(cell_translation)
        data['coords'] = data['frac_coords'] @ graph.lattice_matrix
        supercell_graph.add_node(node, **data)
        
    for edge in graph.edges(data=True):
        for cell_idx in range(np.prod(size)):
            cell_translation1 = np.unravel_index(cell_idx, size)
            cell_translation2 = (edge[2]['cell_translation'] + np.array(cell_translation1)).astype(int)
            if np.all(cell_translation2 >= 0) and np.all(cell_translation2 < size):
                new_source = np.ravel_multi_index(list(cell_translation2) + [edge[0]], size_)
                new_target = np.ravel_multi_index(list(cell_translation1) + [edge[1]], size_)
                data = deepcopy(edge[2])
                supercell_graph.add_edge(new_source, new_target, **data)
        
    setattr(supercell_graph, 'lattice_matrix', graph.lattice_matrix)
    if hasattr(graph, 'spacegroup'):
        setattr(supercell_graph, 'spacegroup', graph.spacegroup)
    return supercell_graph


def to_asymmetric_unit_graph(graph: MultiDiGraph) -> MultiDiGraph:

    asymmetric_mapping = np.array([node[1] for node in graph.nodes(data='asymmetric_mapping')])
    if None in asymmetric_mapping:
        raise ValueError("Graph does not contain symmetry informations. \
Make sure to create the graph with `structure_to_empty_graph` with the `symmetrize` \
argument set to `True`.")
    asu_node_indice, inv_asymmetric_mapping = np.unique(asymmetric_mapping, return_inverse=True)
    
    asu_graph = MultiDiGraph()
    setattr(asu_graph, 'lattice_matrix', graph.lattice_matrix)
    setattr(asu_graph, 'spacegroup', graph.spacegroup)
    new_nodes_idx = {}

    # Add nodes of asymmetric unit to asu_graph
    for i, node_idx in enumerate(asu_node_indice):
        new_nodes_idx[node_idx] = i
        data = deepcopy(graph.nodes[node_idx])
        data['unit_cell_index'] = data['asymmetric_mapping']
        del data['asymmetric_mapping']
        del data['symmop']
        asu_graph.add_node(i, **data)

    if graph.number_of_edges() == 0:
        return asu_graph

    edges_to_keep = graph.in_edges(asu_node_indice, data=True)
    for e in edges_to_keep:
        source_data = graph.nodes[e[0]]
        new_source_idx = inv_asymmetric_mapping[e[0]]
        new_target_idx = new_nodes_idx[e[1]]
        data = deepcopy(e[2])
        data['symmop'] = source_data['symmop']
        asu_graph.add_edge(new_source_idx, new_target_idx, **data)

    return asu_graph

def _to_unit_cell(frac_coords):
    return frac_coords % 1. % 1.

