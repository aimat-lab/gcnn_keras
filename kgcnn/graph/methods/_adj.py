import numpy as np
import scipy.sparse as sp


def precompute_adjacency_scaled(adj_matrix, add_identity: bool = True):
    r"""Precompute the scaled adjacency matrix :math:`A_{s} = D^{-0.5} (A + I) D^{-0.5}`
    after Thomas N. Kipf and Max Welling (2016). Where :math:`I` denotes the diagonal unity matrix.
    The node degree matrix is defined as :math:`D_{i,i} = \sum_{j} (A + I)_{i,j}`.

    Args:
        adj_matrix (np.ndarray, scipy.sparse): Adjacency matrix :math:`A` of shape `(N, N)`.
        add_identity (bool, optional): Whether to add identity :math:`I` in :math:`(A + I)`. Defaults to True.

    Returns:
        array-like: Scaled adjacency matrix after :math:`A_{s} = D^{-0.5} (A + I) D^{-0.5}`.
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = np.array(adj_matrix, dtype="float")
        if add_identity:
            adj_matrix = adj_matrix + np.identity(adj_matrix.shape[0])
        rowsum = np.sum(adj_matrix, axis=-1)
        colsum = np.sum(adj_matrix, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            d_ii = np.power(rowsum, -0.5).flatten()
            d_jj = np.power(colsum, -0.5).flatten()
            d_ii = np.nan_to_num(d_ii, nan=0.0, posinf=0.0, neginf=0.0)
            d_jj = np.nan_to_num(d_jj, nan=0.0, posinf=0.0, neginf=0.0)
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
        with np.errstate(divide='ignore', invalid='ignore'):
            d_ii = np.power(rowsum, -0.5).flatten()
            d_jj = np.power(colsum, -0.5).flatten()
            d_ii = np.nan_to_num(d_ii, nan=0.0, posinf=0.0, neginf=0.0)
            d_jj = np.nan_to_num(d_jj, nan=0.0, posinf=0.0, neginf=0.0)
        di = sp.diags(d_ii, format='coo')
        dj = sp.diags(d_jj, format='coo')
        return di.dot(adj).dot(dj).tocoo()
    else:
        raise TypeError("Matrix format not supported: %s" % type(adj_matrix))


def rescale_edge_weights_degree_sym(edge_indices, edge_weights):
    r"""Normalize edge weights as :math:`\tilde(e)_{i,j} = d_{i,i}^{-0.5} e_{i,j} d_{j,j}^{-0.5}`.
    The node degree is defined as :math:`D_{i,i} = \sum_{j} A_{i, j}`.


    Args:
        edge_indices (np.ndarray): Index-list referring to nodes of shape `(N, 2)`
        edge_weights (np.ndarray): Edge weights matching indices of shape `(N, 1)`

    Returns:
        edge_weights (np.ndarray):  Rescaled edge weights of shape
    """
    if len(edge_indices) == 0:
        return np.array([])
    row_val, row_cnt = np.unique(edge_indices[:, 0], return_counts=True)
    col_val, col_cnt = np.unique(edge_indices[:, 1], return_counts=True)
    d_row = np.zeros(len(edge_weights), dtype=edge_weights.dtype)
    d_col = np.zeros(len(edge_weights), dtype=edge_weights.dtype)
    d_row[row_val] = row_cnt
    d_col[col_val] = col_cnt
    with np.errstate(divide='ignore', invalid='ignore'):
        d_ii = np.power(d_row, -0.5).flatten()
        d_jj = np.power(d_col, -0.5).flatten()
        d_ii = np.nan_to_num(d_ii, nan=0.0, posinf=0.0, neginf=0.0)
        d_jj = np.nan_to_num(d_jj, nan=0.0, posinf=0.0, neginf=0.0)
    new_weights = np.expand_dims(d_ii[edge_indices[:, 0]], axis=-1) * edge_weights * np.expand_dims(
        d_jj[edge_indices[:, 1]], axis=-1)
    return new_weights


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
        a = np.array(adj_scaled > 0, dtype="bool")
        edge_weight = adj_scaled[a]
        index1 = np.tile(np.expand_dims(np.arange(0, a.shape[0]), axis=1), (1, a.shape[1]))
        index2 = np.tile(np.expand_dims(np.arange(0, a.shape[1]), axis=0), (a.shape[0], 1))
        index12 = np.concatenate([np.expand_dims(index1, axis=-1), np.expand_dims(index2, axis=-1)], axis=-1)
        edge_index = index12[a]
        return edge_index, edge_weight
    elif isinstance(adj_scaled, (sp.bsr.bsr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix, sp.csr.csr_matrix)):
        adj_scaled = adj_scaled.tocoo()
        ei1 = np.array(adj_scaled.row.tolist(), dtype="int")
        ei2 = np.array(adj_scaled.col.tolist(), dtype="int")
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


def add_self_loops_to_edge_indices(edge_indices, *args,
                                   remove_duplicates: bool = True, sort_indices: bool = True,
                                   fill_value: int = 0, return_nested: bool = False):
    r"""Add self-loops to edge index list, i.e. `[0, 0], [1, 1], ...]`. Edge values are filled up with ones or zeros.
    Default mode is to remove duplicates in the added list. Edge indices are sorted by default. Sorting is done for the
    first index at position `index[:, 0]`.

    Args:
        edge_indices (np.ndarray): Index-list for edges referring to nodes of shape `(N, 2)`.
        args (np.ndarray): Edge related value arrays to be changed accordingly of shape `(N, ...)`.
        remove_duplicates (bool): Remove duplicate edge indices. Default is True.
        sort_indices (bool): Sort final edge indices. Default is True.
        fill_value (int): Value to initialize edge values with.
        return_nested (bool): Whether to return nested args in addition to indices.

    Returns:
        np.ndarray: `edge_indices` or `(edge_indices, *args)`. Or `(edge_indices, args)` if `return_nested`.
    """
    clean_edge = [x for x in args]
    if len(edge_indices) <= 0:
        if return_nested:
            return edge_indices, clean_edge
        if len(clean_edge) > 0:
            return [edge_indices] + clean_edge
        else:
            return edge_indices
    max_ind = np.max(edge_indices)
    self_loops = np.arange(max_ind + 1, dtype="int")
    self_loops = np.concatenate([np.expand_dims(self_loops, axis=-1), np.expand_dims(self_loops, axis=-1)], axis=-1)
    added_loops = np.concatenate([edge_indices, self_loops], axis=0)
    clean_index = added_loops
    for i, x in enumerate(clean_edge):
        edge_loops_shape = [self_loops.shape[0]] + list(x.shape[1:]) if len(x.shape) > 1 else [
            self_loops.shape[0]]
        edge_loops = np.full(edge_loops_shape, fill_value=fill_value, dtype=x.dtype)
        clean_edge[i] = np.concatenate([x, edge_loops], axis=0)
    if remove_duplicates:
        un, unis = np.unique(clean_index, return_index=True, axis=0)
        mask_all = np.zeros(clean_index.shape[0], dtype="bool")
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
    if return_nested:
        return clean_index, clean_edge
    if len(clean_edge) > 0:
        return [clean_index] + clean_edge
    else:
        return clean_index


def add_edges_reverse_indices(edge_indices, *args, remove_duplicates: bool = True, sort_indices: bool = True,
                              return_nested: bool = False):
    r"""Add matching edges for `(i, j)` as `(j, i)` with the same edge values. If they do already exist,
    no edge is added. By default, all indices are sorted. Sorting is done for the first index at position `index[:, 0]`.

    Args:
        edge_indices (np.ndarray): Index-list of edges referring to nodes of shape `(N, 2)`.
        args (np.ndarray): Edge related value arrays to be changed accordingly of shape `(N, ...)`.
        remove_duplicates (bool): Remove duplicate edge indices. Default is True.
        sort_indices (bool): Sort final edge indices. Default is True.
        return_nested (bool): Whether to return nested args in addition to indices.

    Returns:
        np.ndarray: `edge_indices` or `(edge_indices, *args)`. Or `(edge_indices, args)` if `return_nested`.
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
        mask_all = np.zeros(clean_index.shape[0], dtype="bool")
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
    if return_nested:
        return clean_index, clean_edge
    if len(clean_edge) > 0:
        return [clean_index] + clean_edge
    else:
        return clean_index


def sort_edge_indices(edge_indices, *args, return_nested: bool = False):
    r"""Sort edge index list of `np.ndarray` for the first index and then for the second index.
    Edge values are rearranged accordingly if passed to the function call.

    Args:
        edge_indices (np.ndarray): Edge indices referring to nodes of shape `(N, 2)`.
        args (np.ndarray): Edge related value arrays to be sorted accordingly of shape `(N, ...)`.
        return_nested (bool): Whether to return nested args in addition to indices.

    Returns:
        np.ndarray: `edge_indices` or `(edge_indices, *args)`. Or `(edge_indices, args)` if `return_nested`.
    """
    order1 = np.argsort(edge_indices[:, 1], axis=0, kind='mergesort')  # stable!
    ind1 = edge_indices[order1]
    args1 = [x[order1] for x in args]
    order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')
    ind2 = ind1[order2]
    args2 = [x[order2] for x in args1]
    if return_nested:
        return ind2, args2
    if len(args2) > 0:
        return [ind2] + args2
    else:
        return ind2


def make_adjacency_from_edge_indices(edge_indices, edge_values=None, shape=None):
    r"""Make adjacency as sparse matrix from a list or ``np.ndarray`` of edge_indices and possible values.
    Not for batches, only for single instance.

    Args:
        edge_indices (np.ndarray): List of edge indices of shape `(N, 2)`
        edge_values (np.ndarray): List of possible edge values of shape `(N, )`
        shape (tuple): Shape of the sparse matrix. Default is None.

    Returns:
        scipy.coo.coo_matrix: Sparse adjacency matrix.
    """
    row = np.array(edge_indices[:, 0])
    col = np.array(edge_indices[:, 1])
    if edge_values is None:
        edge_values = np.ones(edge_indices.shape[0])
    if shape is None:
        edi_max = np.max(edge_indices)
        shape = (edi_max + 1, edi_max + 1)
    data = edge_values
    out_adj = sp.coo_matrix((data, (row, col)), shape=shape)
    return out_adj


def get_angle_indices(idx, check_sorted: bool = True, allow_multi_edges: bool = False,
                      allow_self_edges: bool = False, allow_reverse_edges: bool = False,
                      edge_pairing: str = "jk"):
    r"""Compute index list for edge-pairs forming an angle. Not for batches, only for single instance.

    Args:
        idx (np.ndarray): List of edge indices referring to nodes of shape `(N, 2)`
        check_sorted (bool): Whether to sort for new angle indices. Default is True.
        allow_self_edges (bool): Whether to allow the exact same edge in an angle pairing. Overrides multi and reverse
            edge checking.
        allow_multi_edges (bool): Whether to keep angle pairs with same node indices,
            such as angle pairings of sort `ij`, `ij`.
        allow_reverse_edges (bool): Whether to keep angle pairs with reverse node indices,
            such as angle pairings of sort `ij`, `ji`.
        edge_pairing (str): Determines which edge pairs for angle computation are chosen. Default is 'jk'.
            Alternatives are for example: 'ik', 'kj', 'ki', where 'k' denotes the variable index as 'i', 'j' are fixed.

    Returns:
        tuple: idx, idx_ijk, idx_ijk_ij

        - idx (np.ndarray): Original edge indices referring to nodes of shape `(N, 2)`.
        - idx_ijk (np.ndarray): Indices of nodes forming an angle as (i ,j, k) of shape `(M, 3)`.
        - idx_ij_jk (np.ndarray): Indices for edge pairs referring to angles of shape `(M, 2)`.
    """
    if idx is None:
        return None, None, None
    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([])
    # Labeling edges.
    label_ij = np.expand_dims(np.arange(len(idx)), axis=-1)
    # Find edge pairing indices.
    if "k" not in edge_pairing:
        raise ValueError("Edge pairing must have index 'k'.")
    if "i" not in edge_pairing and "j" not in edge_pairing:
        raise ValueError("Edge pairing must have at least one fix index 'i' or 'j'.")
    pos_k = 0 if edge_pairing[0] == "k" else 1
    pos_fix = 0 if edge_pairing[0] != "k" else 1
    pos_ij = 0 if "i" in edge_pairing else 1

    idx_ijk = []  # index triples that form an angle as (i, j, k)
    idx_ij_k = []  # New indices that refer to edges to form an angle as ij, `edge_pairing`
    for n, ij in enumerate(idx):
        matching_edges = idx
        matching_labels = label_ij

        # Condition to find matching xk or kx.
        mask = matching_edges[:, pos_fix] == ij[pos_ij]

        if not allow_multi_edges:
            mask = np.logical_and(mask, np.logical_or(matching_edges[:, 0] != ij[0], matching_edges[:, 1] != ij[1]))

        if not allow_reverse_edges:
            mask = np.logical_and(mask, np.logical_or(matching_edges[:, 0] != ij[1], matching_edges[:, 1] != ij[0]))

        if allow_self_edges:
            mask[n] = True
        else:
            mask[n] = False

        matching_edges, matching_labels = matching_edges[mask], matching_labels[mask]  # apply mask

        if len(matching_edges) == 0:
            idx_ijk.append(np.empty((0, 3), dtype=idx.dtype))
            idx_ij_k.append(np.empty((0, 2), dtype=idx.dtype))
            continue

        # All combos for edge ij
        combos_ik = np.concatenate(
            [np.repeat([ij], len(matching_edges), axis=0), np.expand_dims(matching_edges[:, pos_k], axis=-1)], axis=-1)
        combos_label = np.concatenate(
            [np.repeat([[n]], len(matching_labels), axis=0), matching_labels], axis=-1)
        idx_ijk.append(combos_ik)
        idx_ij_k.append(combos_label)

    idx_ijk = np.concatenate(idx_ijk, axis=0)
    idx_ij_k = np.concatenate(idx_ij_k, axis=0)

    if check_sorted:
        order1 = np.argsort(idx_ij_k[:, 1], axis=0, kind='mergesort')  # stable!
        idx_ij_k = idx_ij_k[order1]
        idx_ijk = idx_ijk[order1]
        order2 = np.argsort(idx_ij_k[:, 0], axis=0, kind='mergesort')
        idx_ijk = idx_ijk[order2]
        idx_ij_k = idx_ij_k[order2]

    return idx, idx_ijk, idx_ij_k


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
    if coord is None or indices is None:
        return
    if len(indices) == 0:
        return np.array([])
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


def coordinates_to_distancematrix(coord3d: np.ndarray):
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


def distance_to_gauss_basis(inputs, bins: int = 20, distance: float = 4.0, sigma: float = 0.4, offset: float = 0.0,
                            axis: int = -1, expand_dims: bool = True):
    r"""Convert distance array to smooth one-hot representation using Gaussian functions.
    Changes shape for Gaussian distance expansion from `(..., )` to (..., bins) by default.

    Note: The default values match realistic units in Angstrom for atoms or molecules.

    Args:
        inputs (np.ndarray): Array of distances of shape `(..., )`.
        bins (int): Number of bins to sample distance from. Default is 20.
        distance (value): Maximum distance to be captured by bins. Default is 4.0.
        sigma (value): Sigma of the Gaussian function, determining the width/sharpness. Default is 0.4.
        offset (float): Possible offset to center Gaussian. Default is 0.0.
        axis (int): Axis to expand distance. Defaults to -1.
        expand_dims (bool): Whether to expand dims. Default to True.

    Returns:
        np.ndarray: Array of Gaussian distance with expanded last axis `(..., #bins)`
    """
    gamma = 1 / sigma / sigma * (-1) / 2
    if expand_dims:
        inputs = np.expand_dims(inputs, axis=axis)
    gauss_bins = np.arange(0, bins, 1) / bins * distance
    expanded_shape = [1]*len(inputs.shape)
    expanded_shape[axis] = len(gauss_bins)
    gauss_bins = np.broadcast_to(gauss_bins, expanded_shape)
    output = np.square(inputs - gauss_bins - offset) * gamma  # (N,M,...,1) - (1,1,...,GBins)
    return np.exp(output)


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

            - graph_adjacency (np.array): Adjacency Matrix of shape `(..., N, N)` of type `bool`.
            - graph_indices (np.array): Flatten indices from former array that have `True` as entry in the
                returned adjacency matrix.
    """
    distance_matrix = np.array(distance_matrix)
    num_atoms = distance_matrix.shape[-1]
    if exclusive:
        graph_adjacency = np.ones_like(distance_matrix, dtype="bool")
    else:
        graph_adjacency = np.zeros_like(distance_matrix, dtype="bool")
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
        temp = np.zeros_like(distance_matrix, dtype="bool")
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


def compute_reverse_edges_index_map(edge_idx: np.ndarray):
    r"""Computes the index map of the reverse edge for each of the edges if available. This can be used by a model
    to directly select the corresponding edge of :math:`(j, i)` which is :math:`(i, j)`.

    Edges that do not have a reverse pair get a `-2147483648` as map index.
    If there are multiple edges, the first encounter is assigned.

    Args:
        edge_idx (np.ndarray): Array of edge indices of shape `(N, 2)`.

    Returns:
        np.ndarray: Map of reverse indices of shape `(N, )`.
    """
    if len(edge_idx) == 0:
        return np.array([], dtype="int")

    edge_idx_rev = np.flip(edge_idx, axis=-1)
    edge_pos, rev_pos = np.where(
        np.all(np.expand_dims(edge_idx, axis=1) == np.expand_dims(edge_idx_rev, axis=0), axis=-1))
    # May have duplicates, find unique.
    ege_pos_uni, uni_pos = np.unique(edge_pos, return_index=True)
    rev_pos_uni = rev_pos[uni_pos]
    edge_map = np.empty(len(edge_idx), dtype="int")
    edge_map.fill(np.iinfo(edge_map.dtype).min)
    edge_map[ege_pos_uni] = rev_pos_uni
    return edge_map
