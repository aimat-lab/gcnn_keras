import warnings
from copy import deepcopy, copy
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from networkx import MultiDiGraph
from pyxtal import pyxtal
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.optimization.neighbors import find_points_in_spheres
from typing import Union, Optional, Any


def get_symmetrized_graph(structure: Union[Structure, pyxtal]) -> MultiDiGraph:
    """Builds a unit graph without any edges, but with symmetry information as node attributes.

    Each node has a `asymmetric_mapping` attribute,
    which contains the id of the symmetry-equivalent atom in the asymmetric unit.
    Each node has a `symmop` attribute,
    which contains the affine matrix to generate the position (in fractional coordinates) of the atom,
    from its symmetry-equivalent atom position in the asymmetric unit.
    Each node has a `multiplicity` attribute,
    which contains the multiplicity of the atom (how many symmetry-equivalent atoms there are for this node).
    The resulting graph will have a `spacegroup` attribute, that specifies the spacegroup of the crystal.

    Args:
        structure (Union[Structure, pyxtal]): Crystal structure to convert to a graph.

    Raises:
        ValueError: If the argument is not a pymatgen Structure or pyxtal object.

    Returns:
        MultiDiGraph: Unit graph with symmetry information, but without any edges for the crystal.
    """
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
        raise ValueError("This method takes either a pymatgen.core.structure.Structure or a pyxtal object.")

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
                       symmop=symmops[node_idx],
                       multiplicity=multiplicities[node_idx])
    setattr(graph, 'lattice_matrix', lattice)
    setattr(graph, 'spacegroup', pyxtal_cell.group.number)

    return graph


def structure_to_empty_graph(structure: Union[Structure, pyxtal], symmetrize: bool = False) -> MultiDiGraph:
    """Builds an unit graph without any edges.

    Args:
        structure (Union[Structure, pyxtal]): Crystal structure to convert to a graph.
            symmetrize (bool, optional): Whether to include symmetry information attributes
            (`asymmetric_mapping`, `symmop`, `multiplicity` attributes) in nodes and graph
            (`spacegroup` atribute).
            Defaults to False.
        symmetrize (bool): Whether to get symmetrized graph.

    Raises:
        ValueError: If the argument is not a pymatgen Structure or pyxtal object.

    Returns:
        MultiDiGraph: Unit graph without any edges for the crystal.
    """
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
                           coords=site.coords,
                           **site.properties)
        setattr(graph, 'lattice_matrix', structure.lattice.matrix)
        return graph


def add_knn_bonds(graph: MultiDiGraph, k: int = 12, max_radius: float = 10.,
                  tolerance: Optional[float] = None, inplace: bool = False) -> MultiDiGraph:
    """Adds kNN-based edges to a unit cell graph.

    Args:
        graph (MultiDiGraph): The unit cell graph to add kNN-based edges to.
        k (int, optional): How many neighbors to add for each node. Defaults to 12.
        max_radius (float, optional): This parameter has no effect on the outcome of the graph.
            It may only on the runtime.
            The algorithm starts the kNN search in the environment the radius of max_radius.
            If the kth neighbor is not within this radius the algorithm is called again with twice the initial radius.
            Defaults to 10.
        tolerance (Optional[float], optional): If tolerance is not None,
            edges with distances of the k-th nearest neighbor plus the tolerance value are included in the graph.
            Defaults to None.
        inplace (bool, optional): Whether to add the edges to the given graph or create a copy with added edges.
            Defaults to False.

    Returns:
        MultiDiGraph: Graph with added edges.
    """
    lattice = _get_attr_from_graph(graph, "lattice_matrix", make_copy=True)
    frac_coords = np.array([data[1] for data in graph.nodes(data='frac_coords')])
    coords = frac_coords @ lattice
    if max_radius is None:
        max_radius = _estimate_nn_radius_from_density(k, coords, lattice, 0.1)
    # return coords, lattice
    index1, index2, offset_vectors, distances = find_points_in_spheres(
        coords,
        coords,
        r=max_radius,
        pbc=np.array([True] * 3, dtype=int),
        lattice=lattice,
        tol=1e-8
    )
    offset_vectors = offset_vectors.astype('i2')
    # Remove self_loops:
    no_self_loops = np.argwhere(~np.isclose(distances, 0)).reshape(-1)
    index1 = index1[no_self_loops]
    index2 = index2[no_self_loops]
    offset_vectors = offset_vectors[no_self_loops]
    distances = distances[no_self_loops]

    new_graph = graph if inplace else deepcopy(graph)

    for node_idx in range(new_graph.number_of_nodes()):
        idxs = np.argwhere(index1 == node_idx)[:, 0]
        sorted_idxs = idxs[np.argsort(distances[idxs])]
        if len(sorted_idxs) < k:
            return add_knn_bonds(new_graph, k=k, max_radius=max_radius * 2, inplace=True)

        if tolerance is not None:
            cutoff = distances[sorted_idxs[k - 1]] + tolerance
            edge_idxs = idxs[np.argwhere(distances[idxs] <= cutoff)][:, 0]
        else:
            edge_idxs = sorted_idxs[:k]
        # If the max_radius doesn't capture k neighbors, try again with double the max_radius
        for edge_idx in edge_idxs:
            new_graph.add_edge(
                index2[edge_idx], index1[edge_idx],
                cell_translation=offset_vectors[edge_idx], distance=distances[edge_idx])

    return new_graph


def add_radius_bonds(graph: MultiDiGraph, radius: float = 5., inplace: bool = False) -> MultiDiGraph:
    """Adds radius-based edges to a unit cell graph.

    Args:
        graph (MultiDiGraph): The unit cell graph to add radius-based edges to.
        radius (float, optional): Cutoff radius for each atom in Angstrom units. Defaults to 5.
        inplace (bool, optional): Whether to add the edges to the given graph or create a copy with added edges.
            Defaults to False.

    Returns:
        MultiDiGraph: Graph with added edges.
    """
    new_graph = graph if inplace else deepcopy(graph)

    lattice = _get_attr_from_graph(graph, "lattice_matrix", make_copy=True)
    frac_coords = np.array([data[1] for data in graph.nodes(data='frac_coords')])
    coords = frac_coords @ lattice
    index1, index2, offset_vectors, distances = find_points_in_spheres(
        coords, coords, r=radius, pbc=np.array([True] * 3, dtype=int), lattice=lattice, tol=1e-8)
    offset_vectors = offset_vectors.astype('i2')
    # Remove self_loops:
    no_self_loops = np.argwhere(~np.isclose(distances, 0)).reshape(-1)
    index1 = index1[no_self_loops]
    index2 = index2[no_self_loops]
    offset_vectors = offset_vectors[no_self_loops]
    distances = distances[no_self_loops]

    if len(index1) == 0:
        warnings.warn(
            'No edges added to the graph, consider increasing the radius and check your graph input instance.')

    for source, target, cell_translation, dist in zip(index2, index1, offset_vectors, distances):
        new_graph.add_edge(source, target, cell_translation=cell_translation, distance=dist)

    return new_graph


def add_voronoi_bonds(graph: MultiDiGraph, min_ridge_area: Optional[float] = None,
                      inplace: bool = False) -> MultiDiGraph:
    """Adds Voronoi-based edges to a unit cell graph.

    Args:
        graph (MultiDiGraph): The unit cell graph to add radius-based edges to.
        min_ridge_area (Optional[float], optional): Threshold value for ridge area between two Voronoi cells.
            If a ridge area between two voronoi cells is smaller than this value the corresponding edge between
            the atoms of the cells is excluded from the graph. Defaults to None.
        inplace (bool, optional): Whether to add the edges to the given graph or create a copy with added edges.
            Defaults to False.

    Returns:
        MultiDiGraph: Graph with added edges.
    """
    new_graph = graph if inplace else deepcopy(graph)

    lattice = _get_attr_from_graph(graph, "lattice_matrix")
    frac_coords = np.array([data[1] for data in graph.nodes(data='frac_coords')])
    dim = lattice.shape[0]
    assert dim == 3
    size = np.array([1, 1, 1])
    expanded_frac_coords = _get_super_cell_grid_frac_coords(lattice, frac_coords, size)
    expanded_coords = expanded_frac_coords @ lattice
    flattened_expanded_coords = expanded_coords.reshape(-1, dim)

    voronoi = Voronoi(flattened_expanded_coords)
    ridge_points_unraveled = np.array(np.unravel_index(voronoi.ridge_points, expanded_coords.shape[:-1]))
    # shape: (num_ridges, 2 (source, target), 4 (3 cell_index + 1 atom_index))
    ridge_points_unraveled = np.moveaxis(ridge_points_unraveled, np.arange(dim), np.roll(np.arange(dim), 1))

    # Filter ridges that have source in the centered unit cell
    source_in_center_cell = np.argwhere(np.all(ridge_points_unraveled[:, 0, :dim] == 1, axis=-1))[:, 0]
    # Filter ridges that have target in the centered unit cell
    target_in_center_cell = np.argwhere(np.all(ridge_points_unraveled[:, 1, :dim] == 1, axis=-1))[:, 0]

    edge_info = np.vstack(
        [ridge_points_unraveled[source_in_center_cell][:, [1, 0]], ridge_points_unraveled[target_in_center_cell]])

    cell_translations = (edge_info[:, 0, :-1] - size).astype(float)
    edge_indices = edge_info[:, :, -1]

    distances = []
    for i in range(len(edge_indices)):
        d = np.linalg.norm(expanded_coords[tuple(edge_info[i][0])] - expanded_coords[tuple(edge_info[i][1])])
        distances.append(d)

    if min_ridge_area is not None:
        ridge_vertices = [voronoi.ridge_vertices[i] for i in
                          np.concatenate([source_in_center_cell, target_in_center_cell])]
        ridge_areas = [get_ridge_area(voronoi.vertices[idxs]) for idxs in ridge_vertices]
        for nodes, cell_translation, dist, ridge_area in zip(edge_indices, cell_translations, distances, ridge_areas):
            source, target = nodes[0], nodes[1]
            if ridge_area > min_ridge_area:
                new_graph.add_edge(source, target, cell_translation=cell_translation, distance=dist,
                                   voronoi_ridge_area=ridge_area)
    else:
        for nodes, cell_translation, dist in zip(edge_indices, cell_translations, distances):
            source, target = nodes[0], nodes[1]
            new_graph.add_edge(source, target, cell_translation=cell_translation, distance=dist)

    return new_graph


def remove_duplicate_edges(graph: MultiDiGraph, inplace=False) -> MultiDiGraph:
    """Removes duplicate edges with same offset.

    Args:
        graph (MultiDiGraph): The unit cell graph with edges to remove.
        inplace (bool, optional): Whether to add the edges to the given graph or create a copy with added edges.
            Defaults to False.

    Returns:
        MultiDiGraph: The graph without duplicate edges.
    """
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


def prune_knn_bonds(graph: MultiDiGraph, k: int = 12, tolerance: Optional[float] = None,
                    inplace: bool = False) -> MultiDiGraph:
    """Prunes edges of a graph to only the k with the smallest distance value.

    Args:
        graph (MultiDiGraph): The unit cell graph with edges to prune.
        k (int, optional): How many neighbors each node should maximally have. Defaults to 12.
        tolerance (Optional[float], optional): If tolerance is not None,
            edges with distances of the k-th nearest neighbor plus the tolerance value are included in the graph.
            Defaults to None.
        inplace (bool, optional): Whether to add the edges to the given graph or create a copy with added edges.
            Defaults to False.

    Returns:
        MultiDiGraph: The graph with pruned edges.
    """
    new_graph = graph if inplace else deepcopy(graph)

    delete_edges = []
    for n in new_graph:
        edges = list(new_graph.in_edges(n, data='distance', keys=True))
        edges.sort(key=lambda x: x[3])
        if tolerance is not None:
            radius = edges[k][3] + tolerance
            delete_edges += [e[:3] for e in edges if e[3] > radius]
        else:
            delete_edges += [e[:3] for e in edges[k:]]
    new_graph.remove_edges_from(delete_edges)
    return new_graph


def prune_radius_bonds(graph: MultiDiGraph, radius: float = 4., inplace: bool = False) -> MultiDiGraph:
    """Prunes edges of a graph with larger distance than the specified radius.

    Args:
        graph (MultiDiGraph): The unit cell graph with edges to prune.
        radius (float, optional): Distance threshold. Edges with larger distance than this value are
            removed from the graph. Defaults to 4.
        inplace (bool, optional): Whether to add the edges to the given graph or create a copy with added edges.
            Defaults to False.

    Returns:
        MultiDiGraph: The graph with pruned edges.
    """
    new_graph = graph if inplace else deepcopy(graph)

    delete_edges = []
    for e in new_graph.edges(data='distance', keys=True):
        if e[3] > radius:
            delete_edges.append(e[:3])
    new_graph.remove_edges_from(delete_edges)
    return new_graph


def prune_voronoi_bonds(graph: MultiDiGraph, min_ridge_area: Optional[float] = None,
                        inplace: bool = False) -> MultiDiGraph:
    """Prunes edges of a graph with a voronoi ridge are smaller then the specified min_ridge_area.

    Only works for graphs with edges that contain `voronoi_ridge_area` as edge attributes.
    Args:
        graph (MultiDiGraph): The unit cell graph with edges to prune.
        min_ridge_area (Optional[float], optional): Threshold value for ridge area between two Voronoi cells.
                If a ridge area between two voronoi cells is smaller than this value the corresponding edge between
                the atoms of the cells is excluded from the graph. Defaults to None.
        inplace (bool, optional): Whether to add the edges to the given graph or create a copy with added edges.
            Defaults to False.

    Returns:
        MultiDiGraph: The graph with pruned edges.
    """
    new_graph = graph if inplace else deepcopy(graph)

    if min_ridge_area is None:
        return new_graph

    delete_edges = []
    for e in new_graph.edges(data='voronoi_ridge_area', keys=True):
        if e[3] < min_ridge_area:
            delete_edges.append(e[:3])
    new_graph.remove_edges_from(delete_edges)
    return new_graph


def add_edge_information(graph: MultiDiGraph, inplace=False,
                         frac_offset=False, offset=True, distance=True) -> MultiDiGraph:
    """Adds edge information, such as offset ( `frac_offset`, `offset` ) and distances ( `distance` ) to edges.

    Args:
        graph (MultiDiGraph): Graph for which to add edge information.
        inplace (bool, optional): Whether to add the edge information to the given graph
            or create a copy with added edges.
            Defaults to False.
        frac_offset (bool, optional): Whether to add fractional offsets (`frac_offset` attribute) to edges.
            Defaults to False.
        offset (bool, optional): Whether to add offsets (`offset` attribute) to edges.
            Defaults to True.
        distance (bool, optional): Whether to add distances (`distance` attribute) to edges.
            Defaults to True.

    Returns:
        MultiDiGraph: The graph with added edge information.
    """
    new_graph = graph if inplace else deepcopy(graph)
    if graph.number_of_edges() == 0:
        return new_graph

    add_frac_offset = frac_offset
    add_offset = offset
    add_distance = distance

    frac_coords1 = []
    frac_coords2 = []
    cell_translations = []

    # Collect necessary coordinate information for calculations
    for e in new_graph.edges(data='cell_translation'):
        frac_coords1.append(new_graph.nodes[e[0]]['frac_coords'])
        cell_translations.append(e[2])
        frac_coords2.append(new_graph.nodes[e[1]]['frac_coords'])

    # Do calculations in vectorized form (instead of doing it inside the edge loop)
    frac_coords1 = np.array(frac_coords1)
    frac_coords2 = np.array(frac_coords2)
    cell_translations = np.array(cell_translations)
    frac_offset = frac_coords2 - (frac_coords1 + cell_translations)
    offset = frac_offset @ _get_attr_from_graph(new_graph, "lattice_matrix")
    if add_distance:
        distances = np.linalg.norm(offset, axis=-1)
    else:
        distances = None

    # Add calculated information to edge attributes
    for i, e in enumerate(new_graph.edges(data=True)):
        if add_frac_offset:
            e[2]['frac_offset'] = frac_offset[i]
        if add_offset:
            e[2]['offset'] = offset[i]
        if add_distance:
            e[2]['distance'] = distances[i]

    return new_graph


def to_non_periodic_unit_cell(graph: MultiDiGraph, add_reverse_edges: bool = True,
                              inplace: bool = False) -> MultiDiGraph:
    """Generates non-periodic graph representation from unit cell graph representation.

    Args:
        graph (MultiDiGraph): Unit cell graph to generate non-periodic graph for.
        add_reverse_edges (bool, optional): Whether to add incoming edges to atoms
            that lie outside the central unit cell.
            Defaults to True.
        inplace (bool, optional): Whether to add distances (`distance` attribute) to edges.
            Defaults to False.

    Returns:
        MultiDiGraph: Corresponding non-periodic graph for the given unit cell graph.
    """
    new_graph = graph if inplace else deepcopy(graph)
    new_nodes = dict()
    new_edges = []
    delete_edges = []
    node_counter = new_graph.number_of_nodes()
    for e in new_graph.edges(data=True, keys=True):
        cell_translation = e[3]['cell_translation']
        if np.any(cell_translation != 0):
            node_key = (e[0],) + tuple(cell_translation)
            if node_key not in new_nodes.keys():
                node_attrs = copy(new_graph.nodes[e[0]])
                node_attrs['frac_coords'] = new_graph.nodes[e[0]]['frac_coords'] + cell_translation
                node_attrs['coords'] = node_attrs['frac_coords'] @ _get_attr_from_graph(new_graph, "lattice_matrix")
                new_nodes[node_key] = (node_counter, node_attrs)
                node_counter += 1
            node_number, _ = new_nodes[node_key]
            edge_attrs1 = deepcopy(e[3])
            new_edges.append((node_number, e[1], edge_attrs1))
            if add_reverse_edges:
                edge_attrs2 = deepcopy(e[3])
                if 'frac_offset' in e[3].keys():
                    edge_attrs2['frac_offset'] = -e[3]['frac_offset']
                if 'offset' in e[3].keys():
                    edge_attrs2['offset'] = -e[3]['offset']
                new_edges.append((e[1], node_number, edge_attrs2))
            delete_edges.append((e[0], e[1], e[2]))
    new_graph.remove_edges_from(delete_edges)
    for node_number, node_attrs in new_nodes.values():
        new_graph.add_node(node_number, **node_attrs)
    for e in new_edges:
        new_graph.add_edge(e[0], e[1], **e[2])
    return new_graph


def to_supercell_graph(graph: MultiDiGraph, size) -> MultiDiGraph:
    """Generates super-cell graph representation from unit cell graph representation.

    Args:
        graph (MultiDiGraph): Unit cell graph to generate super cell graph for.
        size (list): How many cells the crystal will get expanded into each dimension.

    Returns:
        MultiDiGraph: Corresponding super cell graph for the given unit cell graph.
    """

    supercell_graph = MultiDiGraph()
    size_ = list(size) + [graph.number_of_nodes()]
    new_num_nodes = np.prod(size_)
    for node in range(new_num_nodes):
        idx = np.unravel_index(node, size_)
        cell_translation = idx[:3]
        node_num = idx[3]
        data = deepcopy(graph.nodes[node_num])
        data['frac_coords'] = data['frac_coords'] + np.array(cell_translation)
        data['coords'] = data['frac_coords'] @ _get_attr_from_graph(graph, "lattice_matrix")
        supercell_graph.add_node(node, **data)

    for edge in graph.edges(data=True):
        for cell_idx in range(np.prod(size)):
            cell_translation1 = np.unravel_index(cell_idx, size)
            cell_translation2 = (edge[2]['cell_translation'] + np.array(cell_translation1)).astype(int)
            if np.all(cell_translation2 >= 0) and np.all(cell_translation2 < size):
                new_source = np.ravel_multi_index(list(cell_translation2) + [edge[0]], size_)
                new_target = np.ravel_multi_index(list(cell_translation1) + [edge[1]], size_)
                data = deepcopy(edge[2])
                # del data['cell_translation']
                supercell_graph.add_edge(new_source, new_target, **data)

    setattr(supercell_graph, 'lattice_matrix', _get_attr_from_graph(graph, "lattice_matrix"))
    if hasattr(graph, 'spacegroup'):
        setattr(supercell_graph, 'spacegroup', graph.spacegroup)
    return supercell_graph


def to_asymmetric_unit_graph(graph: MultiDiGraph) -> MultiDiGraph:
    """Generates super cell graph representation from unit cell graph representation.

    Args:
        graph (MultiDiGraph): Unit cell graph to generate asymmetric unit graph for.

    Returns:
        MultiDiGraph: Corresponding asymmetric unit graph for the given unit cell graph.
    """

    asymmetric_mapping = np.array([node[1] for node in graph.nodes(data='asymmetric_mapping')])
    if None in asymmetric_mapping:
        raise ValueError(
            "".join([
                "Graph does not contain symmetry information. ",
                "Make sure to create the graph with `structure_to_empty_graph` ",
                "with the `symmetrize` argument set to `True` ."
            ])
        )
    asu_node_indice, inv_asymmetric_mapping = np.unique(asymmetric_mapping, return_inverse=True)

    asu_graph = MultiDiGraph()
    setattr(asu_graph, 'lattice_matrix', _get_attr_from_graph(graph, "lattice_matrix"))
    setattr(asu_graph, 'spacegroup', _get_attr_from_graph(graph, "spacegroup"))
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
    r"""Converts fractional coords to be within the :math:`[0,1)` interval.

    Args:
        frac_coords: Fractional coordinates to map into :math:`[0,1)` interval.

    Returns:
        Fractional coordinates within the [0,1) interval.
    """
    return frac_coords % 1. % 1.


def get_ridge_area(ridge_points):
    """Computes the ridge area given ridge points.

    Beware that this function, assumes that the ridge points are (roughly) within a flat subspace plane
    in the 3 dimensional space.
    It computes the area of the convex hull of the points in three dimensions and then divides it by two,
    since both sides of the flat convex hull are included.

    Args:
        ridge_points (np.ndarray): Ridge points to calculate area for.

    Returns:
        float: Ridge area for the given points.
    """
    while ridge_points.shape[0] <= 3:
        # Append copy of points to avoid QHull Error
        ridge_points = np.append(ridge_points, np.expand_dims(ridge_points[0], 0), 0)
    area = ConvexHull(ridge_points, qhull_options='QJ').area / 2
    return area


def pairwise_diff(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Get the pairwise offset difference between two vector sets.

    Args:
        coords1 (np.ndarray): Coordinates of shape (..., n, 3)
        coords2 (np.ndarray): Coordinates of shape (..., m, 3)

    Returns:
        np.ndarray: Difference values of shape (..., n, m, 3)
    """
    # This can be solved more elegantly with normal broadcasting
    # TODO: Check if the same.
    def _reshape_at_axis(arr, axis, new_shape):
        # move reshape axis to last axis position, because np.reshape reshapes this first
        arr_tmp = np.moveaxis(arr, axis, -1)
        shape = arr_tmp.shape[:-1] + new_shape
        new_positions = np.arange(len(new_shape)) + axis
        old_positions = np.arange(len(new_shape)) + (len(arr.shape) - 1)
        # now call np.reshape and move axis to right position
        return np.moveaxis(arr_tmp.reshape(shape), old_positions, new_positions)

    # Difference calculated at last axis of both inputs
    assert coords1.shape[-1] == coords2.shape[-1]
    coords1_reshaped = coords1.reshape(-1, coords1.shape[-1])
    coords2_reshaped = coords2.reshape(-1, coords2.shape[-1])
    diffs = np.expand_dims(coords2_reshaped, 0) - np.expand_dims(coords1_reshaped, 1)
    return _reshape_at_axis(_reshape_at_axis(diffs, 1, coords2.shape[:-1]), 0, coords1.shape[:-1])


def _get_mesh(size: Union[int, list, tuple], dim: int) -> np.ndarray:
    """Utility function to create a numpy mesh grid with indices at last dimension.

    Args:
        size (int, list): Size of each dimension.
        dim (int): Dimension of the grid.

    Returns:
        np.ndarray: Mesh grid of form:
    """
    if isinstance(size, int):
        size = [size] * dim
    else:
        size = list(size)
    assert len(size) == dim

    mesh = np.array(np.meshgrid(*tuple([np.arange(i) for i in size])))
    mesh = np.moveaxis(mesh, [0, 1], [-1, 1])
    return mesh


def _get_cube(dim: int) -> np.ndarray:
    """Generate a cubic mesh.

    Args:
        dim (int): Dimension for cubic mesh.

    Returns:
        np.ndarray: Cubic mesh.
    """
    return _get_mesh(2, dim)


def _get_max_diameter(lattice: np.ndarray) -> Union[float, np.ndarray]:
    """Determine the max diameter of a lattice.

    Args:
        lattice (np.ndarray): Lattice matrix.

    Returns:
        np.ndarray: Max diameter of the lattice.
    """
    dim = lattice.shape[0]
    cube = _get_cube(dim)
    max_radius = np.max(np.linalg.norm((cube - 1 / 2) @ lattice, axis=1))
    return max_radius * 2


def _get_super_cell_grid_frac_coords(lattice: np.ndarray, frac_coords: np.ndarray, size: Union[int, list, np.ndarray]):
    """Get frac coordinates for positions in a grid of unit cells that is a cubic super-cell.

    ..code - block:: python

        import numpy as np
        from kgcnn.crystal.graph_builder import _get_super_cell_grid_frac_coords
        coordinates = _get_super_cell_grid_frac_coords(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.5]]),
            np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            [3, 3, 3]
        )
        print(coordinates.shape)  # (7, 7, 7, 2, 3)

    Args:
        lattice (np.ndarray): Lattice matrix.
        frac_coords (np.ndarray): Fractional coordinates of atoms in unit cell.
        size (list): Size of the super-cell in each dimension.

    Returns:
        np.ndarray: List of fractional coordinates of atoms in the super-cell.
    """
    dim = lattice.shape[0]

    if isinstance(size, int):
        size = [size] * dim
    else:
        size = list(size)
    assert len(size) == dim

    doubled_size = np.array(size) * 2 + 1
    mesh = _get_mesh(doubled_size, dim)
    # frac_coords_expanded.shape == (1,1,1,num_atoms,3) (for dim == 3)
    # noinspection PyTypeChecker
    frac_coords_expanded = np.expand_dims(frac_coords, np.arange(dim).tolist())
    # mesh_expanded.shape == (double_size[0], double_size[1], double_size[2], 1, 3) (for dim == 3)
    mesh_expanded = np.expand_dims(mesh - size, -2)
    expanded_frac_coords = mesh_expanded + frac_coords_expanded

    return expanded_frac_coords


def _get_attr_from_graph(graph: MultiDiGraph, attr_name: str, make_copy: bool = False) -> Union[Any, np.ndarray]:
    """Utility function to obtain graph-level information of the underlying crystal.

    Args:
        graph (MultiDiGraph): Networkx graph object.
        attr_name (str): Name of the attribute.
        make_copy (bool): Copy crystal-graph attribute.

    Returns:
        np.ndarray: Crystal information.
    """
    if hasattr(graph, attr_name):
        if make_copy:
            out = deepcopy(getattr(graph, attr_name))
        else:
            out = getattr(graph, attr_name)
    else:
        raise AttributeError("Must attach attribute '%s' of crystal information to networkx graph." % attr_name)
    return out


def _estimate_nn_radius_from_density(k: int, coordinates: np.ndarray, lattice: np.ndarray,
                                     empirical_tol_factor: float = 0.0):
    """Rough estimate of the expected radius to find N nearest neighbours.

    Args:
        k (int): Number of neighbours.
        coordinates (np.ndarray): Coordinates array.
        lattice (np.ndarray): Lattice matrix.
        empirical_tol_factor (float): Tolerance factor for radius.

    Returns:
        float: estimated radius
    """
    volume_unit_cell = np.sum(np.abs(np.cross(lattice[0], lattice[1]) * lattice[2]))
    density_unit_cell = len(coordinates) / volume_unit_cell
    estimated_nn_volume = k / density_unit_cell  # + len(coordinates)/density_unit_cell
    estimated_nn_radius = abs(float(np.cbrt(estimated_nn_volume / np.pi * 3 / 4)))
    estimated_nn_radius = estimated_nn_radius * (1.0 + empirical_tol_factor)
    return estimated_nn_radius


def _get_geometric_properties_of_unit_cell(coordinates: np.ndarray, lattice: np.ndarray):
    """Diameter of a 3D unit cell and other properties.

    Args:
        coordinates (np.ndarray): Coordinates array.
        lattice (np.ndarray): Lattice matrix.

    Returns:
        tuple: (center_unit_cell, max_diameter_cell, volume_unit_cell, density_unit_cell)
    """
    # lattice_col = np.transpose(lattice)
    lattice_row = lattice
    center_unit_cell = np.sum(lattice_row, axis=0, keepdims=True) / 2  # (1, 3)
    max_radius_cell = np.amax(np.sqrt(np.sum(np.square(lattice_row - center_unit_cell), axis=-1)))
    max_diameter_cell = 2 * max_radius_cell
    volume_unit_cell = np.sum(np.abs(np.cross(lattice[0], lattice[1]) * lattice[2]))
    density_unit_cell = len(coordinates) / volume_unit_cell
    return center_unit_cell[0], max_diameter_cell, volume_unit_cell, density_unit_cell
