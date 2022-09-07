import numpy as np
from typing import Union


def coulomb_matrix_to_inverse_distance_proton(coulomb_mat: np.ndarray,
                                              unit_conversion: float = 1.0, exponent: float = 2.4):
    r"""Convert a Coulomb matrix back to inverse distancematrix plus atomic number.

    Args:
        coulomb_mat (np.ndarray): Coulomb matrix of shape (...,N,N)
        unit_conversion (float) : Whether to scale units for distance. Default is 1.0.
        exponent (float): Exponent for diagonal elements. Default is 2.4.

    Returns:
        tuple: [inv_dist, z]

            - inv_dist (np.ndarray): Inverse distance Matrix of shape (...,N,N).
            - z (np.ndarray): Atom Number corresponding diagonal as proton number (..., N).
    """
    indslie = np.arange(0, coulomb_mat.shape[-1])
    z = coulomb_mat[..., indslie, indslie]
    z = np.power(2 * z, 1 / exponent)
    a = np.expand_dims(z, axis=len(z.shape) - 1)
    b = np.expand_dims(z, axis=len(z.shape))
    zz = a * b
    c = coulomb_mat / zz
    c[..., indslie, indslie] = 0
    c /= unit_conversion
    z = np.array(np.round(z), dtype=np.int)
    return c, z


def make_rotation_matrix(vector: np.ndarray, angle: float):
    r"""Generate rotation matrix around a given vector with a certain angle.

    Only defined for 3 dimensions explicitly here.

    Args:
        vector (np.ndarray, list): vector of rotation axis (3, ) with (x, y, z).
        angle (value): angle in degrees ° to rotate around.

    Returns:
        np.ndarray: Rotation matrix :math:`R` of shape (3, 3) that performs the rotation for :math:`y = R x`.
    """
    angle = angle / 180.0 * np.pi
    norm = (vector[0] ** 2.0 + vector[1] ** 2.0 + vector[2] ** 2.0) ** 0.5
    direction = vector / norm
    matrix = np.zeros((3, 3))
    matrix[0][0] = direction[0] ** 2.0 * (1.0 - np.cos(angle)) + np.cos(angle)
    matrix[1][1] = direction[1] ** 2.0 * (1.0 - np.cos(angle)) + np.cos(angle)
    matrix[2][2] = direction[2] ** 2.0 * (1.0 - np.cos(angle)) + np.cos(angle)
    matrix[0][1] = direction[0] * direction[1] * (1.0 - np.cos(angle)) - direction[2] * np.sin(angle)
    matrix[1][0] = direction[0] * direction[1] * (1.0 - np.cos(angle)) + direction[2] * np.sin(angle)
    matrix[0][2] = direction[0] * direction[2] * (1.0 - np.cos(angle)) + direction[1] * np.sin(angle)
    matrix[2][0] = direction[0] * direction[2] * (1.0 - np.cos(angle)) - direction[1] * np.sin(angle)
    matrix[1][2] = direction[1] * direction[2] * (1.0 - np.cos(angle)) - direction[0] * np.sin(angle)
    matrix[2][1] = direction[1] * direction[2] * (1.0 - np.cos(angle)) + direction[0] * np.sin(angle)
    return matrix


def rotate_to_principle_axis(coord: np.ndarray):
    r"""Rotate a point-cloud to its principle axis.

    This can be a molecule but also some general data.
    It uses PCA via SVD from :obj:`numpy.linalg.svd`. PCA from scikit uses SVD too (:obj:`scipy.sparse.linalg`).

    .. note::
        The data is centered before SVD but shifted back at the output.

    Args:
        coord (np.array): Array of points forming a pointcloud. Important: coord has shape (N,p)
            where N is the number of samples and p is the feature/coordinate dimension e.g. 3 for x,y,z

    Returns:
        tuple: [R, rotated]

            - R (np.array): Rotation matrix of shape (p, p) if input has (N,p)
            - rotated (np.array): Rotated point-could of coord that was the input.
    """
    centroid_c = np.mean(coord, axis=0)
    sm = coord - centroid_c
    zzt = (np.dot(sm.T, sm))  # Calculate covariance matrix
    u, s, vh = np.linalg.svd(zzt)
    # Alternatively SVD of coord with onyly compute vh but not possible for numpy/scipy.
    rotated = np.dot(sm, vh.T)
    rot_shift = rotated + centroid_c
    return vh, rot_shift


def rigid_transform(a: np.ndarray, b: np.ndarray, correct_reflection: bool = False):
    r"""Rotate and shift point-cloud A to point-cloud B. This should implement Kabsch algorithm.
    May also work for input of shape `(...,N,3)` but is not tested.
    Explanation of Kabsch Algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm
    For further literature:
    https://link.springer.com/article/10.1007/s10015-016-0265-x
    https://link.springer.com/article/10.1007%2Fs001380050048


    .. note::
        The numbering of points of A and B must match; not for shuffled point-cloud.
        This works for 3 dimensions only. Uses SVD.

    Args:
        a (np.ndarray): list of points (N,3) to rotate (and translate)
        b (np.ndarray): list of points (N,3) to rotate towards: A to B, where the coordinates (3) are (x,y,z)
        correct_reflection (bool): Whether to allow reflections or just rotations. Default is False.

    Returns:
        list: [A_rot, R, t]

            - A_rot (np.ndarray): Rotated and shifted version of A to match B
            - R (np.ndarray): Rotation matrix
            - t (np.ndarray): translation from A to B
    """
    a = np.transpose(np.array(a))
    b = np.transpose(np.array(b))
    centroid_a = np.mean(a, axis=1)
    centroid_b = np.mean(b, axis=1)
    am = a - np.expand_dims(centroid_a, axis=1)
    bm = b - np.expand_dims(centroid_b, axis=1)
    h = np.dot(am, np.transpose(bm))
    u, s, vt = np.linalg.svd(h)
    r = np.dot(vt.T, u.T)
    d = np.linalg.det(r)
    if d < 0:
        print("Warning: det(R)<0, det(R)=", d)
        if correct_reflection:
            print("Correcting R...")
            vt[-1, :] *= -1
            r = np.dot(vt.T, u.T)
    bout = np.dot(r, am) + np.expand_dims(centroid_b, axis=1)
    bout = np.transpose(bout)
    t = np.expand_dims(centroid_b - np.dot(r, centroid_a), axis=0)
    t = t.T
    return bout, r, t


def coordinates_from_distance_matrix(distance: np.ndarray, use_center: bool = None, dim: int = 3):
    r"""Compute list of coordinates from a distance matrix of shape `(N, N)`.
    May also work for input of shape `(..., N, N)` but is not tested.
    Uses vectorized Alogrithm:
    http://scripts.iucr.org/cgi-bin/paper?S0567739478000522
    https://www.researchgate.net/publication/252396528_Stable_calculation_of_coordinates_from_distance_information
    no check of positive semi-definite or possible k-dim >= 3 is done here
    performs svd from numpy

    Args:
        distance (np.ndarray): distance matrix of shape (N,N) with Dij = abs(ri-rj)
        use_center (int): which atom should be the center, dafault = None means center of mass
        dim (int): the dimension of embedding, 3 is default

    Return:
        np.ndarray: List of Atom coordinates [[x_1,x_2,x_3],[x_1,x_2,x_3],...]
    """
    distance = np.array(distance)
    dim_in = distance.shape[-1]
    if use_center is None:
        # Take Center of mass (slightly changed for vectorization assuming d_ii = 0)
        di2 = np.square(distance)
        di02 = 1 / 2 / dim_in / dim_in * (2 * dim_in * np.sum(di2, axis=-1) - np.sum(np.sum(di2, axis=-1), axis=-1))
        mat_m = (np.expand_dims(di02, axis=-2) + np.expand_dims(di02, axis=-1) - di2) / 2  # broadcasting
    else:
        di2 = np.square(distance)
        mat_m = (np.expand_dims(di2[..., use_center], axis=-2) + np.expand_dims(di2[..., use_center],
                                                                                axis=-1) - di2) / 2
    u, s, v = np.linalg.svd(mat_m)
    vec = np.matmul(u, np.sqrt(np.diag(s)))  # EV are sorted by default
    dist_out = vec[..., 0:dim]
    return dist_out


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
    center_image = np.zeros(center_dist.shape)
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
    if max_distance is None:
        mask_distance = np.ones_like(dist_sort, dtype="bool")
    else:
        mask_distance = dist_sort <= max_distance + abs(numerical_tol)
    # Based on 'max_neighbours'.
    mask_neighbours = np.zeros_like(dist_sort, dtype="bool")
    if max_neighbours is None:
        max_neighbours = dist_sort.shape[-1]
    mask_neighbours[:, :max_neighbours] = True

    if exclusive:
        mask = np.logical_and(mask_neighbours, mask_distance)
    else:
        mask = np.logical_or(mask_neighbours, mask_distance)

    # Select nodes.
    out_dist = dist_sort[mask]
    out_images = dist_images_sort[mask]
    out_indices = dist_indices_sort[mask]

    return [out_indices, out_images, out_dist]
