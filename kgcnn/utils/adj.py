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


def add_self_loops_to_edge_indices(edge_indices, *args, remove_duplicates=True, sort_indices=True):
    r"""Add self-loops to edge index list, i.e. `[0, 0], [1, 1], ...]`. Edge values are filled up with ones.
    Default mode is to remove duplicates in the added list. Edge indices are sorted by default. Sorting is done for the
    first index at position `index[:, 0]`.

    Args:
        edge_indices (np.ndarray): Index-list for edges referring to nodes of shape `(N, 2)`.
        *args (np.ndarray): Edge related value arrays to be changed accordingly of shape `(N, ...)`.
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


def add_edges_reverse_indices(edge_indices, *args, remove_duplicates=True, sort_indices=True):
    r"""Add matching edges for `(i, j)` as `(j, i)` with the same edge values. If they do already exist,
    no edge is added. By default, all indices are sorted. Sorting is done for the first index at position `index[:, 0]`.

    Args:
        edge_indices (np.ndarray): Index-list of edges referring to nodes of shape `(N, 2)`.
        *args (np.ndarray): Edge related value arrays to be changed accordingly of shape `(N, ...)`.
        remove_duplicates (bool): Remove duplicate edge indices. Default is True.
        sort_indices (bool): Sort final edge indices. Default is True.

    Returns:
        np.ndarray: edge_indices or [edge_indices, *args].
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
    r"""Sort edge index list of ``np.ndarray`` for the first index and then for the second index.
    Edge values are rearranged accordingly if passed to the function call.

    Args:
        edge_indices (np.ndarray): Edge indices referring to nodes of shape `(N, 2)`.
        *args (np.ndarray): Edge related value arrays to be sorted accordingly of shape `(N, ...)`.

    Returns:
        list: [edge_indices, **args] or edge_indices
        
            - edge_indices (np.ndarray): Sorted indices of shape `(N, 2)`.
            - *args (np.ndarray): Edge related arrays to be sorted accordingly of shape `(N, ...)`.
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
    No for batches, only for single instance.

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
    """Compute index list for edge-pairs forming an angle. Requires sorted indices.
    No for batches, only for single instance.

    Args:
        idx (np.ndarray): List of edge indices referring to nodes of shape (N, 2)
        check_sorted (bool): Whether to check inf indices are sorted. Default is True.

    Returns:
        tuple: idx, idx_ijk, idx_ijk_ij

        - idx (np.ndarray): Possibly sorted edge indices referring to nodes of shape (N, 2)
        - idx_ijk (np.ndarray): Indices of nodes forming an angle as i<-j<-k of shape (M, 3)
        - idx_ijk_ij (np.ndarray): Indices for an angle referring to edges of shape (M, 2)
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