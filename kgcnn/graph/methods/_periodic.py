import numpy as np
from typing import Union
from pymatgen.optimization.neighbors import find_points_in_spheres


def range_neighbour_lattice(coordinates: np.ndarray, lattice: np.ndarray,
                            max_distance: Union[float, None] = 4.0,
                            max_neighbours: Union[int, None] = None,
                            self_loops: bool = False,
                            exclusive: bool = True,
                            limit_only_max_neighbours: bool = False,
                            numerical_tol: float = 1e-8,
                            manual_super_cell_radius: float = None,
                            super_cell_tol_factor: float = 0.25,
                            ) -> list:
    r"""Generate range connections for a primitive unit cell in a periodic lattice (vectorized).

    .. code-block:: python

        import numpy as np
        from kgcnn.graph.methods import range_neighbour_lattice

        artificial_lattice = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        artificial_atoms = np.array([[0.1, 0.0, 0.0], [0.5, 0.5, 0.5]])
        out = range_neighbour_lattice(artificial_atoms, artificial_lattice)

        real_lattice = np.array([[-8.71172704, -0., -5.02971843],
                                 [-10.97279872, -0.01635133, 8.94600922],
                                 [-6.5538005, 12.48246168, 1.29207947]])
        real_atoms = np.array([[-24.14652308, 12.46611035, 6.41607351],
                               [-2.09180318, 0., -1.20770325],
                               [0., 0., 0.],
                               [-4.35586352, 0., -2.51485921]])
        out_real = range_neighbour_lattice(real_atoms, real_lattice)

    Args:
        coordinates (np.ndarray): Coordinate of nodes in the central primitive unit cell.
        lattice (np.ndarray): Lattice matrix of real space lattice vectors of shape `(3, 3)`.
            The lattice vectors must be given in rows of the matrix!
        max_distance (float, optional): Maximum distance to allow connections, can also be None. Defaults to 4.0.
        max_neighbours (int, optional): Maximum number of allowed neighbours for each central atom. Default is None.
        self_loops (bool, optional): Allow self-loops between the same central node. Defaults to False.
        exclusive (bool): Whether both distance and maximum neighbours must be fulfilled. Default is True.
        limit_only_max_neighbours (bool): Not used.
        numerical_tol  (float): Numerical tolerance for distance cut-off. Default is 1e-8.
        manual_super_cell_radius (float): Not used.
        super_cell_tol_factor (float): Tolerance to increase for search for neighbours. Default is 0.5.

    Returns:
        list: [indices, images, dist]
    """
    # Require either max_distance or max_neighbours to be specified.
    if max_distance is None and max_neighbours is None:
        raise ValueError("Need to specify either `max_distance` or `max_neighbours` or both.")

    # Volume and density of unit cell based on lattice matrix.
    volume_unit_cell = np.sum(np.abs(np.cross(lattice[0], lattice[1]) * lattice[2]))
    density_unit_cell = len(coordinates) / volume_unit_cell

    # Estimated real-space radius for max_neighbours based on density and volume of a single unit cell.
    if max_neighbours is not None:
        estimated_nn_volume = max_neighbours/density_unit_cell  # + len(coordinates)/ density_unit_cell
        estimated_nn_radius = abs(float(np.cbrt(estimated_nn_volume / np.pi * 3 / 4)))
        estimated_nn_radius = estimated_nn_radius
    else:
        estimated_nn_radius = None

    # Determine the required size of super-cell
    if manual_super_cell_radius is not None:
        if max_neighbours is None:
            # Does not make sense to specify manual supercell in this case.
            radius = max_distance
        else:
            radius = abs(manual_super_cell_radius)
    elif max_distance is None:
        radius = estimated_nn_radius
    elif max_neighbours is None:
        radius = max_distance
    else:
        if exclusive:
            radius = min(max_distance, estimated_nn_radius)
        else:
            radius = max(max_distance, estimated_nn_radius)

    _max_iter_nn_test = 100
    _iter_required = 0
    index1, index2, offset_vectors, distances = None, None, None, None
    for i in range(0, _max_iter_nn_test):
        _iter_required = i
        index1, index2, offset_vectors, distances = find_points_in_spheres(
            coordinates, coordinates,
            r=radius, pbc=np.array([True] * 3, dtype=int), lattice=lattice, tol=numerical_tol)
        offset_vectors = offset_vectors.astype('i2')

        # Remove self_loops:
        if not self_loops:
            no_self_loops = np.argwhere(~np.isclose(distances, 0)).reshape(-1)
            index1 = index1[no_self_loops]
            index2 = index2[no_self_loops]
            offset_vectors = offset_vectors[no_self_loops]
            distances = distances[no_self_loops]

        # We always sort here.
        def reorder(order, *args):
            return [x[order] for x in args]

        def limit_to(c, k, r, id1, id2, ov, dd, logical_reduce=None):
            splits = np.cumsum(c)[:-1]
            id1_split = np.split(id1, splits)
            id2_split = np.split(id2, splits)
            ov_split = np.split(ov, splits)
            dd_split = np.split(dd, splits)
            mask_r, mask_k = None, None
            if k is not None:
                mask_k = []
                for x in id1_split:
                    mask_per_node = np.zeros((len(x)), dtype="bool")
                    mask_per_node[:k] = True
                    mask_k.append(mask_per_node)
            if r is not None:
                mask_r = []
                for x in dd_split:
                    mask_per_node = x <= r + + abs(numerical_tol)
                    mask_r.append(mask_per_node)
            if k is not None and r is not None:
                mask = [logical_reduce(x1, x2) for x1, x2 in zip(mask_r, mask_k)]
            elif k is None:
                mask = mask_r
            else:
                mask = mask_k
            out_split = []
            for a in [id1_split, id2_split, ov_split, dd_split]:
                out_split.append(np.concatenate([x[mask[i_node]] for i_node, x in enumerate(a)], axis=0))
            return out_split

        sort_index = np.argsort(distances, kind="stable")
        index1, index2, offset_vectors, distances = reorder(sort_index, index1, index2, offset_vectors, distances)
        sort_index = np.argsort(index1, kind="stable")
        index1, index2, offset_vectors, distances = reorder(sort_index, index1, index2, offset_vectors, distances)

        # Case: radius cutoff. Only consider radius here.
        if max_neighbours is None:
            break

        unique_centers, counts = np.unique(index1, return_counts=True)
        enough_nn = not (np.any(counts < max_neighbours) if len(counts) > 0 else 0 < max_neighbours)
        # Case: knn.
        if max_distance is None:
            if not enough_nn:
                radius = radius * (1.0 + super_cell_tol_factor)
                continue
            else:
                index1, index2, offset_vectors, distances = limit_to(
                    counts, max_neighbours, None, index1, index2, offset_vectors, distances)
                break

        enough_distance = radius >= max_distance
        # Case mixed both radius and knn.
        if exclusive:
            if enough_nn or enough_distance:
                index1, index2, offset_vectors, distances = limit_to(
                    counts, max_neighbours, max_distance, index1, index2, offset_vectors, distances,
                    logical_reduce=np.logical_and)
                break
            else:
                radius = radius * (1.0 + super_cell_tol_factor)
                continue
        else:
            if not enough_nn or not enough_distance:
                radius = radius * (1.0 + super_cell_tol_factor)
                continue
            else:
                index1, index2, offset_vectors, distances = limit_to(
                    counts, max_neighbours, max_distance, index1, index2, offset_vectors, distances,
                    logical_reduce=np.logical_or)
                break

    if _iter_required + 1 >= _max_iter_nn_test:
        raise ValueError("Exceeded maximum number of allowed range extensions for neighbour calculation.")

    out_indices = np.concatenate([np.expand_dims(index1, axis=-1), np.expand_dims(index2, axis=-1)], axis=-1)
    return [out_indices, offset_vectors, distances]


# This is a python/numpy function to find neighbours in a periodic lattice.
# The pymatgen version is preferred, since this version can get slow for very skew lattice matrices.
def range_neighbour_lattice_python_vectorized(
        coordinates: np.ndarray, lattice: np.ndarray,
        max_distance: Union[float, None] = 4.0,
        max_neighbours: Union[int, None] = None,
        self_loops: bool = False,
        exclusive: bool = True,
        limit_only_max_neighbours: bool = False,
        numerical_tol: float = 1e-8,
        manual_super_cell_radius: float = None,
        super_cell_tol_factor: float = 0.25,
        ) -> list:
    r"""Generate range connections for a primitive unit cell in a periodic lattice (vectorized).

    The function generates a supercell of required radius and computes connections of neighbouring nodes
    from the primitive centered unit cell. For :obj:`max_neighbours` the supercell radius is estimated based on
    the unit cell density. Always the smallest necessary supercell is generated based on :obj:`max_distance` and
    :obj:`max_neighbours`. If a supercell for radius :obj:`max_distance` should always be generated but limited by
    :obj:`max_neighbours`, you can set :obj:`limit_only_max_neighbours` to `True`.

    .. warning::

        All atoms should be projected back into the primitive unit cell before calculating the range connections.

    .. note::

        For periodic structure, setting :obj:`max_distance` and :obj:`max_neighbours` to `inf` would also lead
        to an infinite number of neighbours and connections. If :obj:`exclusive` is set to `False`, having either
        :obj:`max_distance` or :obj:`max_neighbours` set to `inf`, will result in an infinite number of neighbours.
        If set to `None`, :obj:`max_distance` or :obj:`max_neighbours` can selectively be ignored.

    .. code-block:: python

        import numpy as np
        from kgcnn.graph.methods._periodic import range_neighbour_lattice_python_vectorized

        artificial_lattice = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        artificial_atoms = np.array([[0.1, 0.0, 0.0], [0.5, 0.5, 0.5]])
        out = range_neighbour_lattice_python_vectorized(artificial_atoms, artificial_lattice)

        real_lattice = np.array([[-8.71172704, -0., -5.02971843],
                                 [-10.97279872, -0.01635133, 8.94600922],
                                 [-6.5538005, 12.48246168, 1.29207947]])
        real_atoms = np.array([[-24.14652308, 12.46611035, 6.41607351],
                               [-2.09180318, 0., -1.20770325],
                               [0., 0., 0.],
                               [-4.35586352, 0., -2.51485921]])
        out_real = range_neighbour_lattice_python_vectorized(real_atoms, real_lattice)

    Args:
        coordinates (np.ndarray): Coordinate of nodes in the central primitive unit cell.
        lattice (np.ndarray): Lattice matrix of real space lattice vectors of shape `(3, 3)`.
            The lattice vectors must be given in rows of the matrix!
        max_distance (float, optional): Maximum distance to allow connections, can also be None. Defaults to 4.0.
        max_neighbours (int, optional): Maximum number of allowed neighbours for each central atom. Default is None.
        self_loops (bool, optional): Allow self-loops between the same central node. Defaults to False.
        exclusive (bool): Whether both distance and maximum neighbours must be fulfilled. Default is True.
        limit_only_max_neighbours (bool): Whether to only use :obj:`max_neighbours` to limit the number of neighbours
            but not use it to calculate super-cell. Requires :obj:`max_distance` to be not `None`.
            Can be used if the super-cell should be generated with certain :obj:`max_distance`. Default is False.
        numerical_tol  (float): Numerical tolerance for distance cut-off. Default is 1e-8.
        manual_super_cell_radius (float): Manual radius for supercell. This is otherwise automatically set by either
            :obj:`max_distance` or :obj:`max_neighbours` or both. For manual supercell only. Default is None.
        super_cell_tol_factor (float): Tolerance factor for supercell relative to unit cell size. Default is 0.25.

    Returns:
        list: [indices, images, dist]
    """
    # Require either max_distance or max_neighbours to be specified.
    if max_distance is None and max_neighbours is None:
        raise ValueError("Need to specify either `max_distance` or `max_neighbours` or both.")

    # Here we set the lattice matrix, with lattice vectors in either columns or rows of the matrix.
    lattice_col = np.transpose(lattice)
    lattice_row = lattice

    # Index list for nodes. Enumerating the nodes in the central unit cell.
    node_index = np.expand_dims(np.arange(0, len(coordinates)), axis=1)  # Nx1

    # Diagonals, center, volume and density of unit cell based on lattice matrix.
    center_unit_cell = np.sum(lattice_row, axis=0, keepdims=True) / 2  # (1, 3)
    max_radius_cell = np.amax(np.sqrt(np.sum(np.square(lattice_row - center_unit_cell), axis=-1)))
    max_diameter_cell = 2*max_radius_cell
    volume_unit_cell = np.sum(np.abs(np.cross(lattice[0], lattice[1]) * lattice[2]))
    density_unit_cell = len(node_index) / volume_unit_cell

    # Center cell distance. Compute the distance matrix separately for the central primitive unit cell.
    # Here one can check if self-loops (meaning loops between the nodes of the central cell) should be allowed.
    center_indices = np.indices((len(node_index), len(node_index)))
    center_indices = center_indices.transpose(np.append(np.arange(1, 3), 0))  # NxNx2
    center_dist = np.expand_dims(coordinates, axis=0) - np.expand_dims(coordinates, axis=1)  # NxNx3
    center_image = np.zeros(center_dist.shape, dtype="int")
    if not self_loops:
        def remove_self_loops(x):
            m = np.logical_not(np.eye(len(x), dtype="bool"))
            x_shape = np.array(x.shape)
            x_shape[1] -= 1
            return np.reshape(x[m], x_shape)
        center_indices = remove_self_loops(center_indices)
        center_image = remove_self_loops(center_image)
        center_dist = remove_self_loops(center_dist)

    # Check the maximum atomic distance, since in practice atoms may not be inside the unit cell. Although they SHOULD
    # be projected back into the cell.
    max_diameter_atom_pair = np.amax(center_dist) if len(coordinates) > 1 else 0.0
    max_distance_atom_origin = np.amax(np.sqrt(np.sum(np.square(coordinates), axis=-1)))

    # Mesh Grid list. For a list of indices bounding left and right make a list of a 3D mesh.
    # Function is used to pad image unit cells or their origin for super-cell.
    def mesh_grid_list(bound_left: np.array, bound_right: np.array) -> np.array:
        pos = [np.arange(i, j+1, 1) for i, j in zip(bound_left, bound_right)]
        grid_list = np.array(np.meshgrid(*pos)).T.reshape(-1, 3)
        return grid_list

    # Estimated real-space radius for max_neighbours based on density and volume of a single unit cell.
    if max_neighbours is not None:
        estimated_nn_volume = (max_neighbours + len(node_index)) / density_unit_cell
        estimated_nn_radius = abs(float(np.cbrt(estimated_nn_volume / np.pi * 3 / 4)))
    else:
        estimated_nn_radius = None

    # Determine the required size of super-cell
    if manual_super_cell_radius is not None:
        super_cell_radius = abs(manual_super_cell_radius)
    elif max_distance is None:
        super_cell_radius = estimated_nn_radius
    elif max_neighbours is None or limit_only_max_neighbours:
        super_cell_radius = max_distance
    else:
        if exclusive:
            super_cell_radius = min(max_distance, estimated_nn_radius)
        else:
            super_cell_radius = max(max_distance, estimated_nn_radius)

    # Safety for super-cell radius. We add this distance to ensure that all atoms of the outer images are within the
    # actual cutoff distance requested.
    super_cell_tolerance = max(max_diameter_cell, max_diameter_atom_pair, max_distance_atom_origin)
    super_cell_tolerance *= (1.0 + super_cell_tol_factor)

    # Bounding box of real space cube with edge length 2 or inner sphere of radius 1 transformed into index
    # space gives 'bounding_box_unit'. Simply upscale for radius of super-cell.
    # To account for node pairing within the unit cell we add 'max_diameter_cell'.
    bounding_box_unit = np.sum(np.abs(np.linalg.inv(lattice_col)), axis=1)
    bounding_box_index = bounding_box_unit * (super_cell_radius + super_cell_tolerance)
    bounding_box_index = np.ceil(bounding_box_index).astype("int")

    # Making grid for super-cell that repeats the unit cell for required indices in 'bounding_box_index'.
    # Remove [0, 0, 0] of center unit cell by hand.
    bounding_grid = mesh_grid_list(-bounding_box_index, bounding_box_index)
    bounding_grid = bounding_grid[
        np.logical_not(np.all(bounding_grid == np.array([[0, 0, 0]]), axis=-1))]  # Remove center cell
    bounding_grid_real = np.dot(bounding_grid, lattice_row)

    # Check which centers are in the sphere of cutoff, since for non-rectangular lattice vectors, the parallelepiped
    # can be overshooting the required sphere. Better do this here, before computing coordinates of nodes.
    dist_centers = np.sqrt(np.sum(np.square(bounding_grid_real), axis=-1))
    mask_centers = dist_centers <= (super_cell_radius + super_cell_tolerance + abs(numerical_tol))
    images = bounding_grid[mask_centers]
    shifts = bounding_grid_real[mask_centers]

    # Compute node coordinates of images and prepare indices for those nodes. For 'N' nodes per cell and 'C' images
    # (without the central unit cell), this will be (flatten) arrays of (N*C)x3.
    num_images = images.shape[0]
    images = np.expand_dims(images, axis=0)  # 1xCx3
    images = np.repeat(images, len(coordinates), axis=0)  # NxCx3
    coord_images = np.expand_dims(coordinates, axis=1) + shifts  # NxCx3
    coord_images = np.reshape(coord_images, (-1, 3))  # (N*C)x3
    images = np.reshape(images, (-1, 3))  # (N*C)x3
    indices = np.expand_dims(np.repeat(node_index, num_images), axis=-1)  # (N*C)x1

    # Make distance matrix of central cell to all image. This will have shape Nx(NxC).
    dist = np.expand_dims(coord_images, axis=0) - np.expand_dims(coordinates, axis=1)  # Nx(N*C)x3
    dist_indices = np.concatenate(
        [np.repeat(np.expand_dims(node_index, axis=1), len(indices), axis=1),
         np.repeat(np.expand_dims(indices, axis=0), len(node_index), axis=0)], axis=-1)  # Nx(N*C)x2
    dist_images = np.repeat(np.expand_dims(images, axis=0), len(node_index), axis=0)  # Nx(N*C)x3

    # Adding distance matrix of nodes within the central cell to the image distance matrix.
    # The resulting shape is then Nx(NxC+1).
    dist_indices = np.concatenate([center_indices, dist_indices], axis=1)  # Nx(N*C+1)x2
    dist_images = np.concatenate([center_image, dist_images], axis=1)  # Nx(N*C+1)x2
    dist = np.concatenate([center_dist, dist], axis=1)  # Nx(N*C+1)x3

    # Distance in real space.
    dist = np.sqrt(np.sum(np.square(dist), axis=-1))  # Nx(N*C+1)

    # Sorting the distance matrix. Indices and image information must be sorted accordingly.
    arg_sort = np.argsort(dist, axis=-1)
    dist_sort = np.take_along_axis(dist, arg_sort, axis=1)
    dist_indices_sort = np.take_along_axis(
        dist_indices, np.repeat(np.expand_dims(arg_sort, axis=2), dist_indices.shape[2], axis=2), axis=1)
    dist_images_sort = np.take_along_axis(
        dist_images, np.repeat(np.expand_dims(arg_sort, axis=2), dist_images.shape[2], axis=2), axis=1)

    # Select range connections based on distance cutoff and nearest neighbour limit. Uses masking.
    # Based on 'max_distance'.
    mask_distance, mask_neighbours = None, None
    if max_distance is not None:
        mask_distance = dist_sort <= max_distance + abs(numerical_tol)
    # Based on 'max_neighbours'.
    if max_neighbours is not None:
        mask_neighbours = np.zeros_like(dist_sort, dtype="bool")
        mask_neighbours[:, :max_neighbours] = True

    if max_neighbours is None:
        mask = mask_distance
    elif max_distance is None:
        mask = mask_neighbours
    else:
        if exclusive:
            mask = np.logical_and(mask_neighbours, mask_distance)
        else:
            mask = np.logical_or(mask_neighbours, mask_distance)
    # Select nodes.
    out_dist = dist_sort[mask]
    out_images = dist_images_sort[mask]
    out_indices = dist_indices_sort[mask]

    return [out_indices, out_images, out_dist]
