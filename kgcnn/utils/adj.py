import numpy as np
import scipy.sparse as sp


def precompute_adjacency_scaled(adj_matrix, add_identity: bool = True):
    r"""Precompute the scaled adjacency matrix :math:`A_{s} = D^{-0.5} (A + I) D^{-0.5}`
    after Thomas N. Kipf and Max Welling (2016). Where :math:`I` denotes the diagonal unity matrix.
    The node degree matrix is defined as :math:`D_{ii} = \sum_{j} (A + I)_{ij}`.

    Args:
        adj_matrix (np.ndarray, scipy.sparse): Adjacency matrix :math:`A` of shape `(N, N)`.
        add_identity (bool, optional): Whether to add identity :math:`I` in :math:`(A + I)`. Defaults to True.

    Returns:
        array-like: Scaled adjacency matrix after :math:`A_{s} = D^{-0.5} (A + I) D^{-0.5}`.
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = np.array(adj_matrix, dtype=np.float)
        if add_identity:
            adj_matrix = adj_matrix + np.identity(adj_matrix.shape[0])
        rowsum = np.sum(adj_matrix, axis=-1)
        colsum = np.sum(adj_matrix, axis=0)
        d_ii = np.power(rowsum, -0.5).flatten()
        d_jj = np.power(colsum, -0.5).flatten()
        d_ii[np.isinf(d_ii)] = 0.
        d_jj[np.isinf(d_jj)] = 0.
        di = np.zeros((adj_matrix.shape[0], adj_matrix.shape[0]), dtype=adj_matrix.dtype)
        dj = np.zeros((adj_matrix.shape[1], adj_matrix.shape[1]), dtype=adj_matrix.dtype)
        di[np.arange(adj_matrix.shape[0]), np.arange(adj_matrix.shape[0])] = d_ii
        dj[np.arange(adj_matrix.shape[1]), np.arange(adj_matrix.shape[1])] = d_jj
        return np.matmul(di, np.matmul(adj_matrix, dj))
    elif isinstance(adj_matrix, (sp.bsr.bsr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix, sp.csr.csr_matrix)):
        adj = sp.coo_matrix(adj_matrix)
        if add_identity:
            adj = adj + sp.eye(adj.shape[0])
        colsum = np.array(adj.sum(0))
        rowsum = np.array(adj.sum(1))
        d_ii = np.power(rowsum, -0.5).flatten()
        d_jj = np.power(colsum, -0.5).flatten()
        d_ii[np.isinf(d_ii)] = 0.
        d_jj[np.isinf(d_jj)] = 0.
        di = sp.diags(d_ii, format='coo')
        dj = sp.diags(d_jj, format='coo')
        return di.dot(adj).dot(dj).tocoo()
    else:
        raise TypeError("Matrix format not supported: %s" % type(adj_matrix))


def convert_scaled_adjacency_to_list(adj_scaled):
    r"""Map adjacency matrix to index list plus edge weights. In case of a standard adjacency matrix the edge weights
    will be one. For a pre-scaled adjacency matrix they become the entries of :math:`A_{s}`.

    Args:
        adj_scaled (np.ndarray, scipy.sparse): Normal or scaled adjacency matrix :math:`A` of shape `(N, N)`.

    Returns:
        list: [tensor_index, edge_weight]
        
            - tensor_index (np.ndarray): Index-list referring to nodes of shape `(N, 2)`.
            - edge_weight (np.ndarray): Entries of Adjacency matrix of shape `(N, )`.
    """
    if isinstance(adj_scaled, np.ndarray):
        a = np.array(adj_scaled > 0, dtype=np.bool)
        edge_weight = adj_scaled[a]
        index1 = np.tile(np.expand_dims(np.arange(0, a.shape[0]), axis=1), (1, a.shape[1]))
        index2 = np.tile(np.expand_dims(np.arange(0, a.shape[1]), axis=0), (a.shape[0], 1))
        index12 = np.concatenate([np.expand_dims(index1, axis=-1), np.expand_dims(index2, axis=-1)], axis=-1)
        edge_index = index12[a]
        return edge_index, edge_weight
    elif isinstance(adj_scaled, (sp.bsr.bsr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix, sp.csr.csr_matrix)):
        adj_scaled = adj_scaled.tocoo()
        ei1 = np.array(adj_scaled.row.tolist(), dtype=np.int)
        ei2 = np.array(adj_scaled.col.tolist(), dtype=np.int)
        edge_index = np.concatenate([np.expand_dims(ei1, axis=-1), np.expand_dims(ei2, axis=-1)], axis=-1)
        edge_weight = np.array(adj_scaled.data)
        return edge_index, edge_weight
    else:
        raise TypeError("Matrix format not supported: %s." % type(adj_scaled))


def make_adjacency_undirected_logical_or(adj_mat):
    r"""Make adjacency matrix undirected. This adds edges to make adj_matrix symmetric, only if is is not symmetric.
    This is not equivalent to :math:`(A+A^T)/2` but to :math:`A \lor A^T`. This requires the entries of :math:`A` to
    be :math:`\in {0, 1}`.

    Args:
        adj_mat (np.ndarray, scipy.sparse): Adjacency matrix :math:`A` of shape `(N, N)`.

    Returns:
        array-like: Undirected Adjacency matrix. This has :math:`A=A^T`.
    """
    if isinstance(adj_mat, np.ndarray):
        at = np.transpose(adj_mat)
        # Aout = np.logical_or(adj_matrix,at)
        a_out = (at > adj_mat) * at + (adj_mat >= at) * adj_mat
        return a_out
    elif isinstance(adj_mat, (sp.bsr.bsr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix, sp.csr.csr_matrix)):
        adj = sp.coo_matrix(adj_mat)
        adj_t = sp.coo_matrix(adj_mat).transpose()
        a_out = (adj_t > adj).multiply(adj_t) + (adj > adj_t).multiply(adj) + adj - (adj != adj_t).multiply(adj)
        return a_out.tocoo()


def add_self_loops_to_edge_indices(edge_indices, *args, remove_duplicates: bool = True, sort_indices: bool = True):
    r"""Add self-loops to edge index list, i.e. `[0, 0], [1, 1], ...]`. Edge values are filled up with ones.
    Default mode is to remove duplicates in the added list. Edge indices are sorted by default. Sorting is done for the
    first index at position `index[:, 0]`.

    Args:
        edge_indices (np.ndarray): Index-list for edges referring to nodes of shape `(N, 2)`.
        args (np.ndarray): Edge related value arrays to be changed accordingly of shape `(N, ...)`.
        remove_duplicates (bool): Remove duplicate edge indices. Default is True.
        sort_indices (bool): Sort final edge indices. Default is True.

    Returns:
        edge_indices: Sorted index list with self-loops. Optionally (edge_indices, edge_values) if edge_values are not
            None.
    """
    clean_edge = [x for x in args]
    max_ind = np.max(edge_indices)
    self_loops = np.arange(max_ind + 1, dtype=np.int)
    self_loops = np.concatenate([np.expand_dims(self_loops, axis=-1), np.expand_dims(self_loops, axis=-1)], axis=-1)
    added_loops = np.concatenate([edge_indices, self_loops], axis=0)
    clean_index = added_loops
    for i, x in enumerate(clean_edge):
        edge_loops_shape = [self_loops.shape[0]] + list(x.shape[1:]) if len(x.shape) > 1 else [
            self_loops.shape[0]]
        edge_loops = np.ones(edge_loops_shape)
        clean_edge[i] = np.concatenate([x, edge_loops], axis=0)
    if remove_duplicates:
        un, unis = np.unique(clean_index, return_index=True, axis=0)
        mask_all = np.zeros(clean_index.shape[0], dtype=np.bool)
        mask_all[unis] = True
        mask_all[:edge_indices.shape[0]] = True  # keep old indices untouched
        # clean_index = clean_index[unis]
        clean_index = clean_index[mask_all]
        for i, x in enumerate(clean_edge):
            # clean_edge = clean_edge[unis]
            clean_edge[i] = x[mask_all]
    # Sort indices
    if sort_indices:
        order1 = np.argsort(clean_index[:, 1], axis=0, kind='mergesort')  # stable!
        ind1 = clean_index[order1]
        for i, x in enumerate(clean_edge):
            clean_edge[i] = x[order1]
        order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')
        clean_index = ind1[order2]
        for i, x in enumerate(clean_edge):
            clean_edge[i] = x[order2]
    if len(clean_edge) > 0:
        return [clean_index] + clean_edge
    else:
        return clean_index


def add_edges_reverse_indices(edge_indices, *args, remove_duplicates: bool = True, sort_indices: bool = True):
    r"""Add matching edges for `(i, j)` as `(j, i)` with the same edge values. If they do already exist,
    no edge is added. By default, all indices are sorted. Sorting is done for the first index at position `index[:, 0]`.

    Args:
        edge_indices (np.ndarray): Index-list of edges referring to nodes of shape `(N, 2)`.
        args (np.ndarray): Edge related value arrays to be changed accordingly of shape `(N, ...)`.
        remove_duplicates (bool): Remove duplicate edge indices. Default is True.
        sort_indices (bool): Sort final edge indices. Default is True.

    Returns:
        np.ndarray: edge_indices or [edge_indices, args].
    """
    clean_edge = [x for x in args]
    edge_index_flip = np.concatenate([edge_indices[:, 1:2], edge_indices[:, 0:1]], axis=-1)
    edge_index_flip_ij = edge_index_flip[edge_index_flip[:, 1] != edge_index_flip[:, 0]]  # Do not flip self loops
    clean_index = np.concatenate([edge_indices, edge_index_flip_ij], axis=0)
    for i, x in enumerate(clean_edge):
        edge_to_add = x[edge_index_flip[:, 1] != edge_index_flip[:, 0]]
        clean_edge[i] = np.concatenate([x, edge_to_add], axis=0)

    if remove_duplicates:
        un, unis = np.unique(clean_index, return_index=True, axis=0)
        mask_all = np.zeros(clean_index.shape[0], dtype=np.bool)
        mask_all[unis] = True
        mask_all[:edge_indices.shape[0]] = True  # keep old indices untouched
        clean_index = clean_index[mask_all]
        for i, x in enumerate(clean_edge):
            # clean_edge = clean_edge[unis]
            clean_edge[i] = x[mask_all]

    if sort_indices:
        order1 = np.argsort(clean_index[:, 1], axis=0, kind='mergesort')  # stable!
        ind1 = clean_index[order1]
        for i, x in enumerate(clean_edge):
            clean_edge[i] = x[order1]
        order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')
        clean_index = ind1[order2]
        for i, x in enumerate(clean_edge):
            clean_edge[i] = x[order2]
    if len(clean_edge) > 0:
        return [clean_index] + clean_edge
    else:
        return clean_index


def sort_edge_indices(edge_indices, *args):
    r"""Sort edge index list of `np.ndarray` for the first index and then for the second index.
    Edge values are rearranged accordingly if passed to the function call.

    Args:
        edge_indices (np.ndarray): Edge indices referring to nodes of shape `(N, 2)`.
        args (np.ndarray): Edge related value arrays to be sorted accordingly of shape `(N, ...)`.

    Returns:
        list: [edge_indices, args] or edge_indices
        
            - edge_indices (np.ndarray): Sorted indices of shape `(N, 2)`.
            - args (np.ndarray): Edge related arrays to be sorted accordingly of shape `(N, ...)`.
    """
    order1 = np.argsort(edge_indices[:, 1], axis=0, kind='mergesort')  # stable!
    ind1 = edge_indices[order1]
    args1 = [x[order1] for x in args]
    order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')
    ind2 = ind1[order2]
    args2 = [x[order2] for x in args1]
    if len(args2) > 0:
        return [ind2] + args2
    else:
        return ind2


def make_adjacency_from_edge_indices(edge_indices, edge_values=None):
    r"""Make adjacency as sparse matrix from a list or ``np.ndarray`` of edge_indices and possible values.
    Not for batches, only for single instance.

    Args:
        edge_indices (np.ndarray): List of edge indices of shape `(N, 2)`
        edge_values (np.ndarray): List of possible edge values of shape `(N, )`

    Returns:
        scipy.coo.coo_matrix: Sparse adjacency matrix.
    """
    if edge_values is None:
        edge_values = np.ones(edge_indices.shape[0])
    # index_min = np.min(edge_indices)
    index_max = np.max(edge_indices)
    row = np.array(edge_indices[:, 0])
    col = np.array(edge_indices[:, 1])
    data = edge_values
    out_adj = sp.coo_matrix((data, (row, col)), shape=(index_max + 1, index_max + 1))
    return out_adj


def get_angle_indices(idx, check_sorted: bool = True):
    r"""Compute index list for edge-pairs forming an angle. Requires sorted indices.
    Not for batches, only for single instance.

    Args:
        idx (np.ndarray): List of edge indices referring to nodes of shape `(N, 2)`
        check_sorted (bool): Whether to check inf indices are sorted. Default is True.

    Returns:
        tuple: idx, idx_ijk, idx_ijk_ij

        - idx (np.ndarray): Edge indices referring to nodes of shape `(N, 2)`.
        - idx_ijk (np.ndarray): Indices of nodes forming an angle as i<-j<-k of shape `(M, 3)`.
        - idx_ijk_ij (np.ndarray): Indices for an angle referring to edges of shape `(M, 2)`.
    """
    # Verify sorted
    if check_sorted:
        order1 = np.argsort(idx[:, 1], axis=0, kind='mergesort')  # stable!
        ind1 = idx[order1]
        order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')
        ind2 = ind1[order2]
        if not np.array_equal(idx, ind2):
            raise ValueError("ERROR:kgcnn: Indices need to be sorted to compute angles from.")

    pair_label = np.arange(len(idx))
    idx_i = idx[:, 0]
    idx_j = idx[:, 1]
    uni_i, cnts_i = np.unique(idx_i, return_counts=True)
    reps = cnts_i[idx_j]
    idx_ijk_i = np.repeat(idx_i, reps)
    idx_ijk_j = np.repeat(idx_j, reps)
    idx_ijk_label_i = np.repeat(pair_label, reps)
    idx_j_tagged = np.concatenate([np.expand_dims(idx_j, axis=-1), np.expand_dims(pair_label, axis=-1)], axis=-1)
    idx_ijk_k_tagged = np.concatenate([idx_j_tagged[idx_i == x] for x in idx_j])  # This is not fully vectorized
    idx_ijk_k = idx_ijk_k_tagged[:, 0]
    idx_ijk_label_j = idx_ijk_k_tagged[:, 1]
    back_and_forth = idx_ijk_i != idx_ijk_k
    idx_ijk = np.concatenate([np.expand_dims(idx_ijk_i, axis=-1),
                              np.expand_dims(idx_ijk_j, axis=-1),
                              np.expand_dims(idx_ijk_k, axis=-1)], axis=-1)
    idx_ijk = idx_ijk[back_and_forth]
    idx_ijk_ij = np.concatenate([np.expand_dims(idx_ijk_label_i, axis=-1),
                                 np.expand_dims(idx_ijk_label_j, axis=-1)], axis=-1)
    idx_ijk_ij = idx_ijk_ij[back_and_forth]

    return idx, idx_ijk, idx_ijk_ij


def get_angle(coord, indices):
    r"""Compute angle between three points defined by the indices for points i, j, k. Requires mode coordinates.
    With the definition of vector directions :math:`\vec{x}_{ij} = \vec{x}_{i}-\vec{x}_{j}` and
    :math:`\vec{x}_{jk} = \vec{x}_{j}-\vec{x}_{k}`, the angle between for :math:`\vec{x}_{ij}`, :math:`\vec{x}_{jk}`
    is calculated.

    Args:
        coord (np.ndarray): List of coordinates of shape `(N, 3)`.
        indices (np.ndarray): List of indices of shape `(M, 3)`.

    Returns:
        np.ndarray: List of angles matching indices `(M, 1)`.
    """
    xi = coord[indices[:, 0]]
    xj = coord[indices[:, 1]]
    xk = coord[indices[:, 2]]
    v1 = xi - xj
    v2 = xj - xk
    x = np.sum(v1 * v2, axis=-1)
    y = np.cross(v1, v2)
    y = np.linalg.norm(y, axis=-1)
    angle = np.arctan2(y, x)
    angle = np.expand_dims(angle, axis=-1)
    return angle


def get_angle_between_edges(coord, edge_indices, angle_indices):
    r"""Compute angle between two edges that do not necessarily need to be connected by a node.
    However, with the correct choice of angle_indices this can be assured. Node coordinates must be provided.
    The geometric direction of an edge with indices :math:`(i, j)` is given by :math:`\vec{x}_i - \vec{x}_j`.

    Args:
        coord (np.ndarray): List of coordinates of shape `(N, 3)`.
        edge_indices (np.ndarray): List of edge indices referring to node coordinates of shape `(M, 2)`.
        angle_indices (np.ndarray): List of angle indices referring edges of shape `(K, 2)`.

    Returns:
        np.ndarray: List of angles matching angle indices of shape `(K, 1)`.
    """
    xi = coord[edge_indices[:, 0]]
    xj = coord[edge_indices[:, 1]]
    v = xi - xj
    v1 = v[angle_indices[:, 0]]
    v2 = v[angle_indices[:, 1]]
    x = np.sum(v1 * v2, axis=-1)
    y = np.cross(v1, v2)
    y = np.linalg.norm(y, axis=-1)
    angle = np.arctan2(y, x)
    angle = np.expand_dims(angle, axis=-1)
    return angle


def get_index_matrix(shape, flatten=False):
    r"""Matrix of indices with :math:`A_{ijk\dots} = [i,j,k,\dots]` and shape `(N, M, ..., len(shape))`
    with indices being listed in the last dimension.

    Note: Numpy indexing does not work this way but as indices per dimension.

    Args:
        shape (list, int): List of target shape, e.g. (2, 2).
        flatten (bool): Whether to flatten the output or keep input-shape. Default is False.

    Returns:
        np.ndarray: Index array of shape `(N, M, ..., len(shape))`,
            e.g. `[[[0, 0], [0, 1]], [[1, 0], [1, 1]]]` for (2, 2)
    """
    ind_array = np.indices(shape)
    re_order = np.append(np.arange(1, len(shape) + 1), 0)
    ind_array = ind_array.transpose(re_order)
    if flatten:
        ind_array = np.reshape(ind_array, (np.prod(shape), len(shape)))
    return ind_array


def coordinates_to_distancematrix(coord3d):
    r"""Transform coordinates to distance matrix. Will apply transformation on last dimension.
    Changing of shape from `(..., N, 3)` to `(..., N, N)`. This also works for more than 3 coordinates.
    Note: We could extend this to other metrics.

    Arg:
        coord3d (np.ndarray): Coordinates of shape `(..., N, 3)` for cartesian coordinates `(x, y, z)`
            and `N` the number of nodes or points. Coordinates are stored in the last dimension.

    Returns:
        np.ndarray: Distance matrix as numpy array with shape `(..., N, N)` where N is the number of nodes.
    """
    shape_3d = len(coord3d.shape)
    a = np.expand_dims(coord3d, axis=shape_3d - 2)
    b = np.expand_dims(coord3d, axis=shape_3d - 1)
    c = b - a
    d = np.sqrt(np.sum(np.square(c), axis=shape_3d))
    return d


def invert_distance(d, nan=0, pos_inf=0, neg_inf=0):
    r"""Invert distance array, e.g. distance matrix. Inversion is done for all entries.
    Keeps the shape of input distance array, since operation is done element-wise.

    Args:
        d (np.ndarray): Array of distance values of arbitrary shape.
        nan (float): Replacement for np.nan after division. Default is 0.
        pos_inf (float): Replacement for np.inf after division. Default is 0.
        neg_inf (float): Replacement for -np.inf after division. Default is 0.

    Returns:
        np.array: Inverted distance array as np.array of identical shape and
            replaces `np.nan` and `np.inf` with e.g. 0.0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(1, d)
        # c[c == np.inf] = 0
        c = np.nan_to_num(c, nan=nan, posinf=pos_inf, neginf=neg_inf)
    return c


def distance_to_gauss_basis(inputs, bins: int = 20, distance: float = 4.0, sigma: float = 0.4, offset: float = 0.0):
    r"""Convert distance array to smooth one-hot representation using Gaussian functions.
    Changes shape for Gaussian distance expansion from `(..., )` to (..., #bins).
    Note: The default values match realistic units in Angstrom for atoms or molecules.

    Args:
        inputs (np.ndarray): Array of distances of shape `(..., )`.
        bins (int): Number of bins to sample distance from. Default is 20.
        distance (value): Maximum distance to be captured by bins. Default is 4.0.
        sigma (value): Sigma of the Gaussian function, determining the width/sharpness. Default is 0.4.
        offset (float): Possible offset to center Gaussian. Default is 0.0.

    Returns:
        np.ndarray: Array of Gaussian distance with expanded last axis `(..., #bins)`
    """
    gamma = 1 / sigma / sigma * (-1) / 2
    d_shape = inputs.shape
    edge_dist_grid = np.expand_dims(inputs, axis=-1)
    edge_gauss_bin = np.arange(0, bins, 1) / bins * distance
    edge_gauss_bin = np.broadcast_to(edge_gauss_bin, np.append(np.ones(len(d_shape), dtype=np.int32),
                                                               edge_gauss_bin.shape))  # shape (1,1,...,GBins)
    edge_gauss_bin = np.square(edge_dist_grid - edge_gauss_bin - offset) * gamma  # (N,M,...,1) - (1,1,...,GBins)
    edge_gauss_bin = np.exp(edge_gauss_bin)
    return edge_gauss_bin


def define_adjacency_from_distance(distance_matrix, max_distance=np.inf, max_neighbours=np.inf, exclusive=True,
                                   self_loops=False):
    r"""Construct adjacency matrix from a distance matrix by distance and number of neighbours.
    Operates on last axis. Tries to connect nearest neighbours.

    Args:
        distance_matrix (np.array): Distance Matrix of shape `(..., N, N)`
        max_distance (float, optional): Maximum distance to allow connections, can also be None. Defaults to `np.inf`.
        max_neighbours (int, optional): Maximum number of neighbours, can also be None. Defaults to `np.inf`.
        exclusive (bool, optional): Whether both max distance and Neighbours must be fulfilled. Defaults to True.
        self_loops (bool, optional): Allow self-loops on diagonal. Defaults to False.

    Returns:
        tuple: graph_adjacency, graph_indices

        - graph_adjacency (np.array): Adjacency Matrix of shape `(..., N, N)` of type `np.bool`.
        - graph_indices (np.array): Flatten indices from former array that have `True` as entry in the
            returned adjacency matrix.
    """
    distance_matrix = np.array(distance_matrix)
    num_atoms = distance_matrix.shape[-1]
    if exclusive:
        graph_adjacency = np.ones_like(distance_matrix, dtype=np.bool)
    else:
        graph_adjacency = np.zeros_like(distance_matrix, dtype=np.bool)
    inddiag = np.arange(num_atoms)
    # Make Indix Matrix
    indarr = np.indices(distance_matrix.shape)
    re_order = np.append(np.arange(1, len(distance_matrix.shape) + 1), 0)
    graph_indices = indarr.transpose(re_order)
    # print(graph_indices.shape)
    # Add Max Radius
    if max_distance is not None:
        temp = distance_matrix < max_distance
        # temp[...,inddiag,inddiag] = False
        if exclusive:
            graph_adjacency = np.logical_and(graph_adjacency, temp)
        else:
            graph_adjacency = np.logical_or(graph_adjacency, temp)
    # Add #Nieghbours
    if max_neighbours is not None:
        max_neighbours = min(max_neighbours, num_atoms)
        sorting_index = np.argsort(distance_matrix, axis=-1)
        # SortedDistance = np.take_along_axis(self.distance_matrix, sorting_index, axis=-1)
        ind_sorted_red = sorting_index[..., :max_neighbours + 1]
        temp = np.zeros_like(distance_matrix, dtype=np.bool)
        np.put_along_axis(temp, ind_sorted_red, True, axis=-1)
        if exclusive:
            graph_adjacency = np.logical_and(graph_adjacency, temp)
        else:
            graph_adjacency = np.logical_or(graph_adjacency, temp)
    # Allow self-loops
    if not self_loops:
        graph_adjacency[..., inddiag, inddiag] = False

    graph_indices = graph_indices[graph_adjacency]
    return graph_adjacency, graph_indices
