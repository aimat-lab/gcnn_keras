import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.utils.partition import _change_partition_type, _change_edge_tensor_indexing_by_row_partition


class PoolingTopK(ks.layers.Layer):
    """
    Layer for pooling of nodes. Disjoint representation including length tensor.
    
    This implements a learnable score vector plus gate. Implements gPool of Gao et al.
    
    Args:
        k (float): relative number of nodes to remove. Default is 0.1
        kernel_initializer (str): Score initialization. Default is 'glorot_uniform',
        kernel_regularizer (str): Score regularization. Default is None.
        kernel_constraint (bool): Score constrain. Default is None.
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 k=0.1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 partition_type="row_length",
                 node_indexing="batch",
                 **kwargs):
        """Initialize Layer."""
        super(PoolingTopK, self).__init__(**kwargs)
        self.k = k

        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)

        self.partition_type = partition_type
        self.node_indexing = node_indexing

        self.units_p = None
        self.kernel_p = None

    def build(self, input_shape):
        """Build Layer."""
        super(PoolingTopK, self).build(input_shape)

        self.units_p = input_shape[0][-1]
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
            inputs (list): of [nodes, node_partition, edges, edge_partition, edge_indices]

            - nodes (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            - node_partition (tf.tensor): Node length tensor of the numer of nodes
              in each graph of shape (batch,)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Edge length tensor of the numer of edges
              in each graph of shape (batch,)
            - edge_indices (tf.tensor): Flatten edge index list tensor of shape (batch*None,2)
        
        Returns:
            Tuple: [nodes, node_partition, edges, edge_partition, edge_indices],[map_nodes,map_edges]
            
            - nodes (tf.tensor): Pooled node feature tensor of shape (batch*None,F)
            - node_partition (tf.tensor): Pooled node length tensor of the numer of nodes in each graph
              of shape (batch,)
            - edges (tf.tensor): Pooled edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Pooled edge length tensor of the numer of edges in each graph
              of shape (batch,)
            - edge_indices (tf.tensor): Pooled edge index list tensor of shape (batch*None,2) 
            - map_nodes (tf.tensor): Index map between original and pooled nodes (batch*None,)
            - map_edges (tf.tensor): Index map between original and pooled edges of shape (batch*None,)
        """
        node, node_part, edgefeat, edge_part, edgeindref = inputs

        # Determine index dtype
        index_dtype = edgeindref.dtype

        # Make partition tensors
        edgelen = _change_partition_type(edge_part, self.partition_type, "row_length")
        nodelen = _change_partition_type(node_part, self.partition_type, "row_length")

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

        shiftind = _change_edge_tensor_indexing_by_row_partition(edgeindref,
                                                                 nrowlength, edge_ids,
                                                                 partition_type_node="row_length",
                                                                 partition_type_edge="value_rowids",
                                                                 from_indexing=self.node_indexing,
                                                                 to_indexing="batch")

        shiftind = tf.cast(shiftind, dtype=index_dtype)  # already shifted by batch offset (subgraphs)

        # Remove edges that were from filtered nodes via mask
        mask_edge = ks.backend.expand_dims(shiftind, axis=-1) == ks.backend.expand_dims(
            ks.backend.expand_dims(removed_index, axis=0), axis=0)  # this creates large tensor (batch*#edges,2,remove)
        mask_edge = tf.math.logical_not(ks.backend.any(ks.backend.any(mask_edge, axis=-1), axis=-1))
        clean_shiftind = shiftind[mask_edge]
        clean_edge_ids = edge_ids[mask_edge]
        clean_edge_len = tf.math.segment_sum(tf.ones_like(clean_edge_ids), clean_edge_ids)

        # Map edgeindex to new index
        new_edge_index = tf.concat([ks.backend.expand_dims(tf.gather(map_index, clean_shiftind[:, 0]), axis=-1),
                                    ks.backend.expand_dims(tf.gather(map_index, clean_shiftind[:, 1]), axis=-1)],
                                   axis=-1)
        batch_order = tf.argsort(new_edge_index[:, 0], axis=0, direction='ASCENDING', stable=True)
        new_edge_index_sorted = tf.gather(new_edge_index, batch_order, axis=0)

        # Remove the batch offset from edge_indices again for indexing type
        out_indexlist = _change_edge_tensor_indexing_by_row_partition(new_edge_index_sorted,
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
        out_np = _change_partition_type(pooled_len, "row_length", self.partition_type)
        out_ep = _change_partition_type(clean_edge_len, "row_length", self.partition_type)

        # Collect reverse pooling info
        # Remove batch offset for old indicies -> but with new length
        out_pool = _change_edge_tensor_indexing_by_row_partition(pooled_index,
                                                                 nrowlength, pooled_len,
                                                                 partition_type_node="row_length",
                                                                 partition_type_edge="row_length",
                                                                 from_indexing="batch",
                                                                 to_indexing=self.node_indexing,axis=0)
        out_pool_edge = _change_edge_tensor_indexing_by_row_partition(edge_position_new,
                                                                      erowlength, clean_edge_ids,
                                                                      partition_type_node="row_length",
                                                                      partition_type_edge="value_rowids",
                                                                      from_indexing="batch",
                                                                      to_indexing=self.node_indexing,axis=0)

        # Output list
        out = [out_node, out_np, out_edge, out_ep, out_edge_index]
        out_map = [out_pool, out_pool_edge]
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
        config.update({"partition_type": self.partition_type,
                       "node_indexing": self.node_indexing})
        return config


class UnPoolingTopK(ks.layers.Layer):
    """
    Layer for unpooling of nodes. Disjoint representation including length tensor of graphs in batch.
    
    The edge index information are not reverted since the tensor before pooling can be reused. 
    Same holds for batch-assignment in number of nodes and edges information.
    
    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 node_indexing="batch",
                 partition_type="row_length",
                 **kwargs):
        """Initialize Layer."""
        super(UnPoolingTopK, self).__init__(**kwargs)
        self.partition_type = partition_type
        self.node_indexing = node_indexing

    def build(self, input_shape):
        """Build Layer."""
        super(UnPoolingTopK, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_indices, map_node, map_edge, node_pool,
                            node_partition_pool, edge_pool, edge_partition_pool, edge_indices_pool]

            - node (tf.tensor): Original node tensor of shape (batch*None,F_n)
            - node_partition (tf.tensor): Original node partition tensor, e.g. length tensor of shape (batch,)
            - edge (tf.tensor): Original edge feature tensor of shape (batch*None,F_e)
            - edge_partition (tf.tensor): Original edge partition tensor, e.g. length tensor of shape (batch,)
            - edge_indices (tf.tensor): Original index tensor of shape (batch*None,2)
            - map_node (tf.tensor): Index map between original and pooled nodes (batch*None,)
            - map_edge (tf.tensor): Index map between original and pooled edges (batch*None,)
            - node_pool (tf.tensor): Pooled node tensor of shape (batch*None,F_n)
            - node_partition_pool (tf.tensor): Pooled node partition tensor, e.g. length tensor of shape (batch,)
            - edge_pool (tf.tensor): Pooled edge feature tensor of shape (batch*None,F_e)
            - edge_partition_pool (tf.tensor): Pooled edge partition tensor, e.g. length tensor of shape (batch,)
            - edge_indices (tf.tensor): Pooled index tensor of shape (batch*None,2)
        
        Returns:
            List: [nodes, node_partition, edges, edge_partition, edge_indices]
            
            - nodes (tf.tensor): Unpooled node feature tensor of shape (batch*None,F)
            - node_partition (tf.tensor): Unpooled node partition tensor, e.g. length tensor of shape (batch,)
            - edges (tf.tensor): Unpooled edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Unpooled edge partition tensor, e.g. length tensor of shape (batch,)
            - edge_indices (tf.tensor): Unpooled edge index list tensor of shape (batch*None,2)
        """
        node_old, nodepart_old, edge_old, edgepart_old, edgeind_old, map_node, map_edge, node_new, nodpart_new, edge_new, edgepart_new, edgeind_new = inputs

        # Make partition tensors
        nrowlength = _change_partition_type(nodepart_old,self.partition_type,"row_length")
        erowlength =  _change_partition_type(edgepart_old,self.partition_type,"row_length")
        pool_node_len = _change_partition_type(nodpart_new,self.partition_type,"row_length")
        pool_edge_id = _change_partition_type(edgepart_new,self.partition_type,"value_rowids")

        # Correct map index for flatten batch offset
        map_node = _change_edge_tensor_indexing_by_row_partition(map_node,
                                                                  nrowlength, pool_node_len,
                                                                  partition_type_node="row_length",
                                                                  partition_type_edge="row_length",
                                                                  from_indexing=self.node_indexing,
                                                                  to_indexing="batch",axis=0)
        map_edge = _change_edge_tensor_indexing_by_row_partition(map_edge,
                                                                  erowlength, pool_edge_id,
                                                                  partition_type_node="row_length",
                                                                  partition_type_edge="value_rowids",
                                                                  from_indexing=self.node_indexing,
                                                                  to_indexing="batch",axis=0)

        index_dtype = map_node.dtype
        node_shape = tf.stack(
            [tf.cast(tf.shape(node_old)[0], dtype=index_dtype), tf.cast(tf.shape(node_new)[1], dtype=index_dtype)])
        out_node = tf.scatter_nd(ks.backend.expand_dims(map_node, axis=-1), node_new, node_shape)

        index_dtype = map_edge.dtype
        edge_shape = tf.stack(
            [tf.cast(tf.shape(edge_old)[0], dtype=index_dtype), tf.cast(tf.shape(edge_new)[1], dtype=index_dtype)])
        out_edge = tf.scatter_nd(ks.backend.expand_dims(map_edge, axis=-1), edge_new, edge_shape)

        outlist = [out_node, nodepart_old, out_edge, edgepart_old, edgeind_old]
        return outlist

    def get_config(self):
        """Update layer config."""
        config = super(UnPoolingTopK, self).get_config()
        config.update({"partition_type": self.partition_type})
        config.update({"node_indexing": self.node_indexing})
        return config
