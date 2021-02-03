import tensorflow as tf
import tensorflow.keras as ks


class PoolingTopK(ks.layers.Layer):
    """
    Layer for pooling of nodes. Disjoint representation including length tensor.
    
    This implements a learnable score vector plus gate. Implements gPool of Gao et al.
    
    Args:
        k (float): relative number of nodes to remove. Default is 0.1
        kernel_initializer (str): Score initialization. Default is 'glorot_uniform',
        kernel_regularizer (str): Score regularization. Default is None.
        kernel_constraint (bool): Score constrain. Default is None.
        **kwargs
    """
    
    def __init__(self,
                 k = 0.1 ,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer = None,
                 kernel_constraint=None,
                 **kwargs):
        """Initialize Layer."""
        super(PoolingTopK, self).__init__(**kwargs)
        self.k = k
        
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)

    def build(self, input_shape):
        """Build Layer."""
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
        """Forward pass.
        
        Inputs list of [nodes, node_length, edges, edge_length, edge_indices]
            
        Args:
            nodes (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            node_length (tf.tensor): Node length tensor of the numer of nodes
                                     in each graph of shape (batch,)
            edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            edge_length (tf.tensor): Edge length tensor of the numer of edges
                                     in each graph of shape (batch,)
            edge_indices (tf.tensor): Flatten edge index list tensor of shape (batch*None,2)   
        
        Returns:
            Tuple: [nodes, node_length, edges, edge_length, edge_indices],[map_nodes,map_edges]
            
            - nodes (tf.tensor): Pooled node feature tensor of shape (batch*None,F)
            - node_length (tf.tensor): Pooled node length tensor of the numer of nodes in each graph of shape (batch,)
            - edges (tf.tensor): Pooled edge feature list of shape (batch*None,F)
            - edge_length (tf.tensor): Pooled edge length tensor of the numer of edges in each graph of shape (batch,)
            - edge_indices (tf.tensor): Pooled edge index list tensor of shape (batch*None,2) 
            - map_nodes (tf.tensor): Index map between original and pooled nodes (batch*None,)
            - map_edges (tf.tensor): Index map between original and pooled edges of shape (batch*None,)
        """
        node,nodelen,edgefeat,edgelen,edgeind = inputs
        
        #Determine index dtype
        index_dtype = edgeind.dtype
        
        #Get node properties from ragged tensor
        nvalue = node
        nrowlength = tf.cast(nodelen,dtype = index_dtype)
        nids = tf.repeat(tf.range(tf.shape(nrowlength)[0],dtype=index_dtype),nrowlength)
        
        #Use kernel p to get score
        norm_p = ks.backend.sqrt(ks.backend.sum(ks.backend.square(self.kernel_p),axis=-1,keepdims=True))
        nscore = ks.backend.sum(nvalue*self.kernel_p/norm_p,axis=-1)
 
        #Sort nodes according to score
        #Then sort after former node ids -> stable = True keeps previous order
        sort1 = tf.argsort(nscore,direction='ASCENDING', stable=False)
        nids_sorted1 = tf.gather(nids,sort1)
        sort2 = tf.argsort(nids_sorted1,direction='ASCENDING', stable=True) # Must be stable=true here
        sort12 = tf.gather(sort1,sort2)  #index goes from 0 to batch*N, no in batch indexing
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
        shiftind = tf.cast(edgeind,dtype=index_dtype) #already shifted by batch offset (subgraphs)
        edge_ids = tf.repeat(tf.range(tf.shape(edgelen)[0],dtype=index_dtype),edgelen)
        
        #Remove edges that were from filtered nodes via mask
        mask_edge = ks.backend.expand_dims(shiftind,axis=-1) == ks.backend.expand_dims(ks.backend.expand_dims(removed_index,axis=0),axis=0)  #this creates large tensor (batch*#edges,2,remove)
        mask_edge = tf.math.logical_not(ks.backend.any(ks.backend.any(mask_edge,axis=-1),axis=-1))
        clean_shiftind = shiftind[mask_edge]
        clean_edge_ids = edge_ids[mask_edge]
        clean_edge_len = tf.math.segment_sum(tf.ones_like(clean_edge_ids),clean_edge_ids)
        
        # Map edgeindex to new index
        new_edge_index = tf.concat([ks.backend.expand_dims(tf.gather(map_index,clean_shiftind[:,0]),axis=-1),ks.backend.expand_dims(tf.gather(map_index,clean_shiftind[:,1]),axis=-1)],axis=-1)
        batch_order = tf.argsort(new_edge_index[:,0],axis=0,direction='ASCENDING',stable=True)
        new_edge_index_sorted = tf.gather(new_edge_index ,batch_order,axis=0)
        
        #For disjoint representation the batch offset does not need to be removed       
        out_indexlist = new_edge_index_sorted
        
        #Correct edge features the same way (remove and reorder)
        edge_feat = edgefeat
        clean_edge_feat = edge_feat[mask_edge]
        clean_edge_feat_sorted = tf.gather(clean_edge_feat,batch_order,axis=0)
        
        #Make edge feature map for new edge features
        edge_position_old = tf.range(tf.shape(edgefeat)[0],dtype=index_dtype)
        edge_position_new =  edge_position_old[mask_edge]
        edge_position_new = tf.gather(edge_position_new,batch_order,axis=0)
        
        #Collect output tensors
        out_node = gated_n
        out_nlen = pooled_len
        out_edge = clean_edge_feat_sorted
        out_elen = clean_edge_len
        out_edge_index = out_indexlist
        
        #Collect reverse pooling info
        out_pool = pooled_index
        
        out = [out_node,out_nlen,out_edge,out_elen,out_edge_index]
        out_map = [out_pool,edge_position_new]
        return out,out_map
    
    
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
    



class UnPoolingTopK(ks.layers.Layer):
    """
    Layer for unpooling of nodes. Disjoint representation including length tensor of graphs in batch.
    
    The edge index information are not reverted since the tensor before pooling can be reused. 
    Same holds for batch-assignment in number of nodes and edges information.
    
    Args:
        **kwargs
    """
    
    def __init__(self,
                 **kwargs):
        """Initialize Layer."""
        super(UnPoolingTopK, self).__init__(**kwargs)


    def build(self, input_shape):
        """Build Layer."""
        super(UnPoolingTopK, self).build(input_shape)

        
    def call(self, inputs):
        """Forward pass.
        
        Inputs [node, node_length, edge, edge_length, edge_indices, map_node, map_edge, node_pool, node_length_pool, edge_pool, edge_length_pool, edge_indices_pool]
        
        Args:        
            node (tf.tensor): Original node tensor of shape (batch*None,F_n)
            node_length (tf.tensor): Original node length tensor of shape (batch,)
            edge (tf.tensor): Original edge feature tensor of shape (batch*None,F_e)
            edge_length (tf.tensor): Original edge length tensor of shape (batch,)
            edge_indices (tf.tensor): Original index tensor of shape (batch*None,2) 
            map_node (tf.tensor): Index map between original and pooled nodes (batch*None,)
            map_edge (tf.tensor): Index map between original and pooled edges (batch*None,)
            node_pool (tf.tensor): Pooled node tensor of shape (batch*None,F_n)
            node_length_pool (tf.tensor): Pooled node length tensor of shape (batch,)
            edge_pool (tf.tensor): Pooled edge feature tensor of shape (batch*None,F_e)
            edge_length_pool (tf.tensor): Pooled edge length tensor of shape (batch,)
            edge_indices (tf.tensor): Pooled index tensor of shape (batch*None,2) 
        
        Returns:
            List: [nodes, node_length, edges, edge_length, edge_indices]
            
            - nodes (tf.tensor): Unpooled node feature tensor of shape (batch*None,F)
            - node_length (tf.tensor): Unpooled node length tensor of the numer of nodes in each graph of shape (batch,)
            - edges (tf.tensor): Unpooled edge feature list of shape (batch*None,F)
            - edge_length (tf.tensor): Unpooled edge length tensor of the numer of edges in each graph of shape (batch,)
            - edge_indices (tf.tensor): Unpooled edge index list tensor of shape (batch*None,2)
        """
        node_old,nodelen_old,edge_old,edgelen_old,edgeind_old, map_node, map_edge , node_new,nodelen_new,edge_new,edgelen_new,edgeind_new = inputs
        
        index_dtype = map_node.dtype
        node_shape = tf.stack([tf.cast(tf.shape(node_old)[0],dtype=index_dtype),tf.cast(tf.shape(node_new)[1],dtype=index_dtype)])
        out_node = tf.scatter_nd(ks.backend.expand_dims(map_node,axis=-1),node_new,node_shape)
        
        index_dtype = map_edge.dtype
        edge_shape = tf.stack([tf.cast(tf.shape(edge_old)[0],dtype=index_dtype),tf.cast(tf.shape(edge_new)[1],dtype=index_dtype)])
        out_edge = tf.scatter_nd(ks.backend.expand_dims(map_edge,axis=-1),edge_new,edge_shape)

        outlist = [out_node,nodelen_old,out_edge,edgelen_old,edgeind_old]
        return outlist