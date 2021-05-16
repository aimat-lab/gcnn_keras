import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_partition_type, kgcnn_ops_change_edge_tensor_indexing_by_row_partition
from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type, kgcnn_ops_check_tensor_type
from kgcnn.layers.base import GraphBaseLayer

class PoolingTopK(GraphBaseLayer):
    """Layer for pooling of nodes. Disjoint representation including length tensor.
    
    This implements a learnable score vector plus gate. Implements gPool of Gao et al.
    
    Args:
        k (float): relative number of nodes to remove. Default is 0.1
        kernel_initializer (str): Score initialization. Default is 'glorot_uniform',
        kernel_regularizer (str): Score regularization. Default is None.
        kernel_constraint (bool): Score constrain. Default is None.
    """

    def __init__(self,
                 k=0.1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        """Initialize Layer."""
        super(PoolingTopK, self).__init__(**kwargs)
        self.k = k
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)

        self.units_p = None
        self.kernel_p = None

    def build(self, input_shape):
        """Build Layer."""
        super(PoolingTopK, self).build(input_shape)

        if self.input_tensor_type == "values_partition":
            self.units_p = input_shape[0][0][-1]
        elif self.input_tensor_type == "ragged":
            self.units_p = input_shape[0][-1]
        else:
            raise NotImplementedError("Error: Not supported input tensor type.")

        self.kernel_p = self.add_weight('score',
                                        shape=[1, self.units_p],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, node_partition, edges, edge_partition, edge_indices]

            - nodes (tf.ragged): Node embeddings of shape (batch, [N], F)
            - edges (tf.ragged): Edge embeddings of shape (batch, [N], F)
            - edge_indices (tf.ragged): Edge index list of shape of shape (batch, [N], 2)
        
        Returns:
            Tuple: [nodes, edges, edge_indices], [map_nodes, map_edges]
            
            - nodes (tf.ragged): Pooled node feature tensor
            - edges (tf.ragged): Pooled edge feature list
            - edge_indices (tf.ragged): Pooled edge index list
            - map_nodes (tf.ragged): Index map between original and pooled nodes
            - map_edges (tf.ragged): Index map between original and pooled edges
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 3)
        # We cast to values here
        node, nodelen = dyn_inputs[0].values, dyn_inputs[0].row_lengths()
        edgefeat, edgelen = dyn_inputs[1].values, dyn_inputs[1].row_lengths()
        edgeindref, _ = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

        index_dtype = edgeindref.dtype
        # Get node properties
        nvalue = node
        nrowlength = tf.cast(nodelen, dtype=index_dtype)
        erowlength = tf.cast(edgelen, dtype=index_dtype)
        nids = tf.repeat(tf.range(tf.shape(nrowlength)[0], dtype=index_dtype), nrowlength)

        # Use kernel p to get score
        norm_p = ks.backend.sqrt(ks.backend.sum(ks.backend.square(self.kernel_p), axis=-1, keepdims=True))
        nscore = ks.backend.sum(nvalue * self.kernel_p / norm_p, axis=-1)

        # Sort nodes according to score
        # Then sort after former node ids -> stable = True keeps previous order
        sort1 = tf.argsort(nscore, direction='ASCENDING', stable=False)
        nids_sorted1 = tf.gather(nids, sort1)
        sort2 = tf.argsort(nids_sorted1, direction='ASCENDING', stable=True)  # Must be stable=true here
        sort12 = tf.gather(sort1, sort2)  # index goes from 0 to batch*N, no in batch indexing
        nvalue_sorted = tf.gather(nvalue, sort12, axis=0)
        nscore_sorted = tf.gather(nscore, sort12, axis=0)

        # Make Mask
        nremove = tf.cast(tf.math.round(self.k * tf.cast(nrowlength, dtype=tf.keras.backend.floatx())),
                          dtype=index_dtype)
        nkeep = nrowlength - nremove
        n_remove_keep = ks.backend.flatten(
            tf.concat([ks.backend.expand_dims(nremove, axis=-1), ks.backend.expand_dims(nkeep, axis=-1)], axis=-1))
        mask_remove_keep = ks.backend.flatten(tf.concat(
            [ks.backend.expand_dims(tf.zeros_like(nremove, dtype=tf.bool), axis=-1),
             ks.backend.expand_dims(tf.ones_like(nkeep, tf.bool), axis=-1)], axis=-1))
        mask = tf.repeat(mask_remove_keep, n_remove_keep)

        # Apply Mask to remove lower score nodes
        pooled_n = nvalue_sorted[mask]
        pooled_score = nscore_sorted[mask]
        pooled_id = nids[mask]  # nids should not have changed by final sorting
        pooled_len = nkeep  # shape=(batch,)
        pooled_index = tf.cast(sort12[mask], dtype=index_dtype)  # the index goes from 0 to N*batch
        removed_index = tf.cast(sort12[tf.math.logical_not(mask)],
                                dtype=index_dtype)  # the index goes from 0 to N*batch

        # Pass through gate
        gated_n = pooled_n * ks.backend.expand_dims(tf.keras.activations.sigmoid(pooled_score), axis=-1)

        # Make index map for new nodes towards old index
        index_new_nodes = tf.range(tf.shape(pooled_index)[0], dtype=index_dtype)
        old_shape = tf.cast(ks.backend.expand_dims(tf.shape(nvalue)[0]), dtype=index_dtype)
        map_index = tf.scatter_nd(ks.backend.expand_dims(pooled_index, axis=-1), index_new_nodes, old_shape)

        # Shift index if necessary
        edge_ids = tf.repeat(tf.range(tf.shape(edgelen)[0], dtype=index_dtype), edgelen)

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeindref,
                                                                          nrowlength, edge_ids,
                                                                          partition_type_node="row_length",
                                                                          partition_type_edge="value_rowids",
                                                                          from_indexing=self.node_indexing,
                                                                          to_indexing="batch")

        shiftind = tf.cast(shiftind, dtype=index_dtype)  # already shifted by batch offset (sub-graphs)

        # Remove edges that were from filtered nodes via mask
        mask_edge = ks.backend.expand_dims(shiftind, axis=-1) == ks.backend.expand_dims(
            ks.backend.expand_dims(removed_index, axis=0), axis=0)  # this creates large tensor (batch*#edges,2,remove)
        mask_edge = tf.math.logical_not(ks.backend.any(ks.backend.any(mask_edge, axis=-1), axis=-1))
        clean_shiftind = shiftind[mask_edge]
        clean_edge_ids = edge_ids[mask_edge]
        # clean_edge_len = tf.math.segment_sum(tf.ones_like(clean_edge_ids), clean_edge_ids)
        clean_edge_len = tf.scatter_nd(tf.expand_dims(clean_edge_ids, axis=-1), tf.ones_like(clean_edge_ids),
                                       tf.cast(tf.shape(erowlength), dtype=index_dtype))

        # Map edgeindex to new index
        new_edge_index = tf.concat([ks.backend.expand_dims(tf.gather(map_index, clean_shiftind[:, 0]), axis=-1),
                                    ks.backend.expand_dims(tf.gather(map_index, clean_shiftind[:, 1]), axis=-1)],
                                   axis=-1)
        batch_order = tf.argsort(new_edge_index[:, 0], axis=0, direction='ASCENDING', stable=True)
        new_edge_index_sorted = tf.gather(new_edge_index, batch_order, axis=0)

        # Remove the batch offset from edge_indices again for indexing type
        out_indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(new_edge_index_sorted,
                                                                               pooled_len, clean_edge_ids,
                                                                               partition_type_node="row_length",
                                                                               partition_type_edge="value_rowids",
                                                                               from_indexing="batch",
                                                                               to_indexing=self.node_indexing)

        # Correct edge features the same way (remove and reorder)
        edge_feat = edgefeat
        clean_edge_feat = edge_feat[mask_edge]
        clean_edge_feat_sorted = tf.gather(clean_edge_feat, batch_order, axis=0)

        # Make edge feature map for new edge features
        edge_position_old = tf.range(tf.shape(edgefeat)[0], dtype=index_dtype)
        edge_position_new = edge_position_old[mask_edge]
        edge_position_new = tf.gather(edge_position_new, batch_order, axis=0)

        # Collect output tensors
        out_node = gated_n
        out_edge = clean_edge_feat_sorted
        out_edge_index = out_indexlist

        # Change length to partition required
        out_np = pooled_len # row_length
        out_ep = clean_edge_len # row_length

        # Collect reverse pooling info
        # Remove batch offset for old indicies -> but with new length
        out_pool = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(pooled_index,
                                                                          nrowlength, pooled_len,
                                                                          partition_type_node="row_length",
                                                                          partition_type_edge="row_length",
                                                                          from_indexing="batch",
                                                                          to_indexing=self.node_indexing, axis=0)
        out_pool_edge = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_position_new,
                                                                               erowlength, clean_edge_ids,
                                                                               partition_type_node="row_length",
                                                                               partition_type_edge="value_rowids",
                                                                               from_indexing="batch",
                                                                               to_indexing=self.node_indexing, axis=0)

        out = [self._kgcnn_map_output_ragged([out_node, out_np], "row_length", 0),
               self._kgcnn_map_output_ragged([out_edge, out_ep], "row_length", 1),
               self._kgcnn_map_output_ragged([out_edge_index, out_ep], "row_length", 2)]
        out_map = [self._kgcnn_map_output_ragged([out_pool, out_np], "row_length", 0),
                   self._kgcnn_map_output_ragged([out_pool_edge, out_ep], "row_length", 1)]

        return out, out_map

    def get_config(self):
        """Update layer config."""
        config = super(PoolingTopK, self).get_config()
        config.update({"k": self.k})
        config.update({
            'kernel_initializer':
                ks.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer':
                ks.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint':
                ks.constraints.serialize(self.kernel_constraint),
        })
        return config


class UnPoolingTopK(GraphBaseLayer):
    """Layer for un-pooling of nodes from PoolingTopK.
    
    The edge index information are not reverted since the tensor before pooling can be reused. 
    Same holds for batch-assignment in number of nodes and edges information.
    """

    def __init__(self, **kwargs):
        """Initialize Layer."""
        super(UnPoolingTopK, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build Layer."""
        super(UnPoolingTopK, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [node, edge, edge_indices, map_node, map_edge, node_pool, edge_pool, edge_indices_pool]

            - node (tf.ragged): Original node tensor
            - edge (tf.ragged): Original edge feature tensor
            - edge_indices (tf.ragged): Original index tensor
            - map_node (tf.ragged): Index map between original and pooled nodes
            - map_edge (tf.ragged): Index map between original and pooled edges
            - node_pool (tf.ragged): Pooled node tensor
            - edge_pool (tf.ragged): Pooled edge feature tensor
            - edge_indices (tf.ragged): Pooled index tensor
        
        Returns:
            List: [nodes, edges, edge_indices]
            
            - nodes (tf.ragged): Unpooled node feature tensor
            - edges (tf.ragged): Unpooled edge feature list
            - edge_indices (tf.ragged): Unpooled edge index
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 8)
        # We cast to values here
        node_old, nrowlength = dyn_inputs[0].values, dyn_inputs[0].row_lengths()
        edge_old, erowlength = dyn_inputs[1].values, dyn_inputs[1].row_lengths()
        edgeind_old, _ = dyn_inputs[2].values, dyn_inputs[2].row_lengths()
        map_node, _ = dyn_inputs[3].values, dyn_inputs[3].row_lengths()
        map_edge, _ = dyn_inputs[4].values, dyn_inputs[4].row_lengths()
        node_new, pool_node_len = dyn_inputs[5].values, dyn_inputs[5].row_lengths()
        edge_new, pool_edge_id = dyn_inputs[6].values, dyn_inputs[6].value_rowids()
        edgeind_new, _ = dyn_inputs[7].values, dyn_inputs[7].row_lengths()

        # Correct map index for flatten batch offset
        map_node = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(map_node,
                                                                          nrowlength, pool_node_len,
                                                                          partition_type_node="row_length",
                                                                          partition_type_edge="row_length",
                                                                          from_indexing=self.node_indexing,
                                                                          to_indexing="batch", axis=0)
        map_edge = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(map_edge,
                                                                          erowlength, pool_edge_id,
                                                                          partition_type_node="row_length",
                                                                          partition_type_edge="value_rowids",
                                                                          from_indexing=self.node_indexing,
                                                                          to_indexing="batch", axis=0)

        index_dtype = map_node.dtype
        node_shape = tf.stack([tf.cast(tf.shape(node_old)[0], dtype=index_dtype),
                               tf.cast(tf.shape(node_new)[1], dtype=index_dtype)])
        out_node = tf.scatter_nd(ks.backend.expand_dims(map_node, axis=-1), node_new, node_shape)

        index_dtype = map_edge.dtype
        edge_shape = tf.stack([tf.cast(tf.shape(edge_old)[0], dtype=index_dtype),
                               tf.cast(tf.shape(edge_new)[1], dtype=index_dtype)])
        out_edge = tf.scatter_nd(ks.backend.expand_dims(map_edge, axis=-1), edge_new, edge_shape)

        outlist = [self._kgcnn_map_output_ragged([out_node, nrowlength], "row_length", 0),
                   self._kgcnn_map_output_ragged([out_edge, erowlength], "row_length", 1),
                   self._kgcnn_map_output_ragged([edgeind_old, erowlength], "row_length", 2)]
        return outlist

    def get_config(self):
        """Update layer config."""
        config = super(UnPoolingTopK, self).get_config()
        return config
