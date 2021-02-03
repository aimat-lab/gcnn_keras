import tensorflow as tf
import tensorflow.keras as ks



class PoolingTopK(ks.layers.Layer):
    """
    Layer for pooling of nodes.
    
    This implements a learnable score vector plus gate. Implements gPool of Gao et al.
    
    Args:
        k (float): relative number of nodes to remove. Default is 0.1
        kernel_initializer (str): Score initialization. Default is 'glorot_uniform',
        kernel_regularizer (str): Score regularization. Default is None.
        kernel_constraint (str): Score constrain. Default is None.
        ragged_validate (bool): To validate output ragged tensor. Defualt is False.
        **kwargs
    """
    
    def __init__(self,
                 k = 0.1 ,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer = None,
                 kernel_constraint=None,
                 node_indexing = 'sample',
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingTopK, self).__init__(**kwargs)
        self.k = k
        self.ragged_validate = ragged_validate
        self.node_indexing = node_indexing
        
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)

        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        """Build layer."""
        super(PoolingTopK, self).build(input_shape)

        self.units_p = input_shape[0][-1]
        self.kernel_p = self.add_weight( 'score',
                                        shape=[1,self.units_p],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)

        
    def call(self, inputs):
        """Forward Pass.
        
        Inputs list of [nodes,edge_index, edges]
        
        Args: 
            nodes (tf.ragged): Node ragged feature tensor of shape (batch,None,F)
            edge_index (tf.ragged): Edge indices ragged tensor of shape (batch,None,2)
            edges (tf.ragged): Edge feature tensor of shape (batch,None,F)
    
        Returns:
            tuple: [nodes,edge_indices,edges],[map_nodes,map_edges]
            
            - nodes (tf.ragged): Pooled node features of shape (batch,None,F)
            - edge_indices (tf.ragged): Pooled edge indices of shape (batch,None,2)
            - edges (tf.ragged):Pooled edge features of shape (batch,None,F_e)
            - map_nodes (tf.ragged): Index map between original and pooled nodes (batch,None)
            - map_edges (tf.ragged): Index map between original and pooled edges (batch,None)
        """
        node,edgeind,edgefeat = inputs
        
        #Determine index dtype
        index_dtype = edgeind.values.dtype
        
        #Get node properties from ragged tensor
        nvalue = node.values
        nrowsplit = tf.cast(node.row_splits,dtype = index_dtype)
        nrowlength = tf.cast(node.row_lengths(),dtype = index_dtype)
        nids = node.value_rowids()
        
        #Use kernel p to get score
        norm_p = ks.backend.sqrt(ks.backend.sum(ks.backend.square(self.kernel_p),axis=-1,keepdims=True))
        nscore = ks.backend.sum(nvalue*self.kernel_p/norm_p,axis=-1)
 
        #Sort nodes according to score
        #Then sort after former node ids -> stable = True keeps previous order
        sort1 = tf.argsort(nscore,direction='ASCENDING', stable=False)
        nids_sorted1 = tf.gather(nids,sort1)
        sort2 = tf.argsort(nids_sorted1,direction='ASCENDING', stable=True) # Must be stable=true here
        sort12 = tf.gather(sort1,sort2)  #index goes from 0 to batch*N
        nvalue_sorted = tf.gather(nvalue,sort12 ,axis=0)
        nscore_sorted = tf.gather(nscore,sort12 ,axis=0)
        
        #Make Mask
        nremove = tf.cast(tf.math.round(self.k * tf.cast(nrowlength,dtype=tf.keras.backend.floatx())),dtype=index_dtype)
        nkeep = nrowlength - nremove
        n_remove_keep = ks.backend.flatten(tf.concat([ks.backend.expand_dims(nremove,axis=-1),ks.backend.expand_dims(nkeep,axis=-1)],axis=-1))
        mask_remove_keep = ks.backend.flatten(tf.concat([ks.backend.expand_dims(tf.zeros_like(nremove,dtype=tf.bool),axis=-1),ks.backend.expand_dims(tf.ones_like(nkeep,tf.bool),axis=-1)],axis=-1))
        mask = tf.repeat(mask_remove_keep,n_remove_keep)
        
        #Apply Mask to remove lower score nodes
        pooled_n = nvalue_sorted[mask]
        pooled_score = nscore_sorted[mask]
        pooled_id = nids[mask]  #nids should not have changed by final sorting
        pooled_len = nkeep # shape=(batch,)
        pooled_index = tf.cast(sort12[mask],dtype = index_dtype) #the index goes from 0 to N*batch
        removed_index = tf.cast(sort12[tf.math.logical_not(mask)],dtype = index_dtype) #the index goes from 0 to N*batch
        
        #Pass through gate
        gated_n = pooled_n *ks.backend.expand_dims(tf.keras.activations.sigmoid(pooled_score),axis=-1)
        
        #Make index map for new nodes towards old index
        index_new_nodes = tf.range(tf.shape(pooled_index)[0],dtype=index_dtype)
        old_shape = tf.cast(ks.backend.expand_dims(tf.shape(nvalue)[0]),dtype=index_dtype)
        map_index = tf.scatter_nd(ks.backend.expand_dims(pooled_index,axis=-1),index_new_nodes,old_shape)
        
        #Shift also edgeindex by batch offset
        if(self.node_indexing == 'batch'):
            shiftind = tf.cast(edgeind,dtype=index_dtype) #already shifted by batch offset (subgraphs)
            edge_ids  = edgefeat.value_rowids()
        elif(self.node_indexing == 'sample'):
            edge_ids = edgeind.value_rowids()
            shift1 = edgeind.values
            shift2 = tf.expand_dims(tf.gather(nrowsplit[:-1],edge_ids),axis=1)
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
            shiftind = tf.cast(shiftind,dtype=index_dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")

        #Remove edges that were from filtered nodes via mask
        mask_edge = ks.backend.expand_dims(shiftind,axis=-1) == ks.backend.expand_dims(ks.backend.expand_dims(removed_index,axis=0),axis=0)  #this creates large tensor (batch*#edges,2,remove)
        mask_edge = tf.math.logical_not(ks.backend.any(ks.backend.any(mask_edge,axis=-1),axis=-1))
        clean_shiftind = shiftind[mask_edge]
        clean_edge_ids = edge_ids[mask_edge]
        #clean_edge_len = tf.math.segment_sum(tf.ones_like(clean_edge_ids),clean_edge_ids)
        
        # Map edgeindex to new index
        new_edge_index = tf.concat([ks.backend.expand_dims(tf.gather(map_index,clean_shiftind[:,0]),axis=-1),ks.backend.expand_dims(tf.gather(map_index,clean_shiftind[:,1]),axis=-1)],axis=-1)
        batch_order = tf.argsort(new_edge_index[:,0],axis=0,direction='ASCENDING',stable=True)
        new_edge_index_sorted = tf.gather(new_edge_index ,batch_order,axis=0)
        
        #Remove the batch offset from edge indices again for ragged tensor output
        if(self.node_indexing == 'batch'):
            out_indexlist = new_edge_index_sorted
        elif(self.node_indexing == 'sample'):
            batch_index_offset = tf.expand_dims(tf.gather(tf.cumsum(pooled_len,exclusive=True),clean_edge_ids),axis=1)
            out_indexlist = new_edge_index_sorted - tf.cast(batch_index_offset ,dtype=index_dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
            
        #Correct edge features the same way (remove and reorder)
        edge_feat = edgefeat.values
        clean_edge_feat = edge_feat[mask_edge]
        clean_edge_feat_sorted = tf.gather(clean_edge_feat,batch_order,axis=0)
        
        #Make edge feature map for new edge features
        edge_position_old = tf.range(tf.shape(edge_feat)[0],dtype=index_dtype)
        edge_position_new =  edge_position_old[mask_edge]
        edge_position_new = tf.gather(edge_position_new,batch_order,axis=0)
        
        #Build ragged tensors again for node, edgeindex and edge
        out_node = tf.RaggedTensor.from_value_rowids(gated_n,pooled_id,validate=self.ragged_validate)
        out_edge_index = tf.RaggedTensor.from_value_rowids(out_indexlist,clean_edge_ids,validate=self.ragged_validate)
        out_edge = tf.RaggedTensor.from_value_rowids(clean_edge_feat_sorted,clean_edge_ids,validate=self.ragged_validate)
        
        # Build map
        map_node = tf.RaggedTensor.from_row_lengths(pooled_index,pooled_len,validate=self.ragged_validate)
        map_edge = tf.RaggedTensor.from_value_rowids(edge_position_new,clean_edge_ids,validate=self.ragged_validate)
        
        
        out_map = [map_node,map_edge]
        out = [out_node,out_edge_index,out_edge]
        return out,out_map
    
    
    def get_config(self):
        """Update layer config."""
        config = super(PoolingTopK, self).get_config()
        config.update({"k": self.k})
        config.update({"ragged_validate": self.ragged_validate})
        config.update({
        'kernel_initializer':
            ks.initializers.serialize(self.kernel_initializer),
        'kernel_regularizer':
            ks.regularizers.serialize(self.kernel_regularizer),
        'kernel_constraint':
            ks.constraints.serialize(self.kernel_constraint),
        })
        return config 
    

class UnPoolingTopK(ks.layers.Layer):
    """
    Layer for unpooling of nodes.
    
    The edge index information are not reverted since the tensor before pooling can be reused. 
    Same holds for batch-assignment in number of nodes and edges information.
    
    Args:
        ragged_validate (bool): To validate ragged output tensor. Default is False.
        **kwargs
    """
    
    def __init__(self,
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(UnPoolingTopK, self).__init__(**kwargs)
        self.ragged_validate = ragged_validate

    def build(self, input_shape):
        """Build layer."""
        super(UnPoolingTopK, self).build(input_shape)

        
    def call(self, inputs):
        """Forward Pass.
        
        Inputs list of [nodes, edge_indices, edges, map_nodes, map_edges, nodes_pool, edge_indices_pool, edges_pool]
        
        Args:
            nodes (tf.ragged): Original node ragged tensor of shape (batch,None,F)
            edge_indices (tf.ragged): Original edge index ragged tensor of shape (batch,None,2)
            edges (tf.ragged): Original edge feature tensor of shape (batch,None,F)
            map_nodes (tf.ragged): Node index map (batch,None)
            map_edges (tf.ragged): Edge index map (batch,None)
            nodes_pool (tf.ragged): Pooled node ragged tensor of shape (batch,None,F)
            edge_indices_pool (tf.ragged): Pooled edge index ragged tensor of shape (batch,None,2)
            edges_pool (tf.ragged): Pooled edge feature tensor of shape (batch,None,F)
    
        Returns:
            list: [nodes,edge_index,edges]
            
            - nodes (tf.ragged): Unpooled node ragged tensor of shape (batch,None,F_n)
            - edge_index (tf.ragged): Unpooled edge index ragged tensor of shape (batch,None,2)
            - edges (tf.ragged): Unpooled edge feature tensor of shape (batch,None,F_e)
        """
        node_old,edgeind_old,edge_old, map_node, map_edge, node_new,edgeind_new,edge_new = inputs
        
        map_node = map_node.values
        map_edge = map_edge.values
        
        index_dtype = map_node.dtype
        node_old_value = node_old.values
        node_new_value = node_new.values
        node_shape = tf.stack([tf.cast(tf.shape(node_old_value)[0],dtype=index_dtype),tf.cast(tf.shape(node_new_value)[1],dtype=index_dtype)])
        out_node_value = tf.scatter_nd(ks.backend.expand_dims(map_node,axis=-1),node_new_value,node_shape)
        out_node = tf.RaggedTensor.from_row_splits(out_node_value,node_old.row_splits,validate=self.ragged_validate)
        
        
        index_dtype = map_edge.dtype
        edge_old_value = edge_old.values
        edge_new_value = edge_new.values
        edge_shape = tf.stack([tf.cast(tf.shape(edge_old_value)[0],dtype=index_dtype),tf.cast(tf.shape(edge_new_value)[1],dtype=index_dtype)])
        out_edge_value = tf.scatter_nd(ks.backend.expand_dims(map_edge,axis=-1),edge_new_value,edge_shape)
        out_edge = tf.RaggedTensor.from_row_splits(out_edge_value,edge_old.row_splits,validate=self.ragged_validate)

        outlist = [out_node,edgeind_old,out_edge]
        return outlist
    
    def get_config(self):
        """Update layer config."""
        config = super(UnPoolingTopK, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        return config