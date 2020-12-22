"""@package: Keras Layers for graph pooling using ragged tensors
@author: Patrick, 
"""

import tensorflow as tf
import tensorflow.keras as ks



class PoolingTopK(ks.layers.Layer):
    """
    Layer for pooling of nodes. Disjoint representation including length tensor of graphs in batch.
    
    Args:
        k : relative number of nodes to remove
        kernel_initializer : 'glorot_uniform',
        kernel_regularizer : None,
        kernel_constraint : None,
        **kwargs
    
    Inputs:
        Node tensor of shape (batch*None,F_n)
        Node length tensor of shape (batch,)
        Edge feature tensor of shape (batch*None,F_e)
        Edge length tensor of shape (batch,)
        Edge index tensor of shape (batch*None,2)   
    
    Outputs:
        Pooled nodes of shape (batch*None,F_n)
        Pooled node length of shape (batch,)
        Pooled edge features of shape (batch*None,F_e)
        Pooled edge length tensor of shape (batch,)
        Pooled edge indices of shape (batch*None,2)
    """
    def __init__(self,
                 k = 0.1 ,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer = None,
                 kernel_constraint=None,
                 **kwargs):
        super(PoolingTopK, self).__init__(**kwargs)
        self.k = k
        
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)

        self._supports_ragged_inputs = True 
    def build(self, input_shape):
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
        node,nodelen,edgefeat,edgelen,edgeind = inputs
        
        #Determine index dtype
        index_dtype = edgeind.dtype
        
        #Get node properties from ragged tensor
        nvalue = node
        nrowlength = tf.cast(nodelen,dtype = index_dtype)
        nids = tf.repeat(ks.backend.arange(nrowlength.shape[0],dtype=index_dtype),nrowlength)
        
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
        index_new_nodes = ks.backend.arange(pooled_index.shape[0],dtype=index_dtype)
        old_shape = tf.cast(ks.backend.expand_dims(nvalue.shape[0]),dtype=index_dtype)
        map_index = tf.scatter_nd(ks.backend.expand_dims(pooled_index,axis=-1),index_new_nodes,old_shape)
        
        #Shift also edgeindex by batch offset
        shiftind = tf.cast(edgeind,dtype=index_dtype) #already shifted by batch offset (subgraphs)
        edge_ids = tf.repeat(ks.backend.arange(edgelen.shape[0],dtype=index_dtype),edgelen)
        
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
        
        #Collect output tensors
        out_node = gated_n
        out_nlen = pooled_len
        out_edge = clean_edge_feat_sorted
        out_elen = clean_edge_len
        out_edge_index = out_indexlist
        
        out = [out_node,out_nlen,out_edge,out_elen,out_edge_index]
        return out
    
    
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
    


class PoolingTopKbyNodeID(ks.layers.Layer):
    """
    Layer for pooling of nodes. Disjoint representation including length tensor of graphs in batch.
    
    Args:
        k : relative number of nodes to remove
        kernel_initializer : 'glorot_uniform',
        kernel_regularizer : None,
        kernel_constraint : None,
        **kwargs
    
    Inputs:
        hidden_x = Node-Features mit Shape (n_nodes, node_feature_dim)
        hidden_e = Edge-Features mit Shape (n_edges, edge_feature_dim)
        hidden_ix = Graph-Indices mit Shape (n_nodes,), also einfach flat
        hidden_ie = Edge-Paare mit Shape (n_edges, 2)
    
    Outputs:
         hidden_x, 
         hidden_ix, 
         hidden_e, 
         hidden_ie,
         pooled_nodes_mask, 
         pooled_edges_mask
    """
    def __init__(self,
                 k = 0.1 ,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer = None,
                 kernel_constraint=None,
                 **kwargs):
        super(PoolingTopKbyNodeID, self).__init__(**kwargs)
        self.k = k
        
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)

        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(PoolingTopKbyNodeID, self).build(input_shape)

        self.units_p = input_shape[0][-1]
        self.kernel_p = self.add_weight( 'score',
                                        shape=[1,self.units_p],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)

        
    def call(self, inputs):
        node,edgefeat,nids,edgeind = inputs
        
        #Determine index dtype
        index_dtype = edgeind.dtype
        
        #Get node properties from ragged tensor
        nvalue = node
        nids = tf.cast(nids,dtype = index_dtype)
        nodelen = tf.math.segment_sum(tf.ones_like(nids,dtype=index_dtype),nids)
        nrowlength = tf.cast(nodelen,dtype = index_dtype)
        
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
        index_new_nodes = ks.backend.arange(pooled_index.shape[0],dtype=index_dtype)
        old_shape = tf.cast(ks.backend.expand_dims(nvalue.shape[0]),dtype=index_dtype)
        map_index = tf.scatter_nd(ks.backend.expand_dims(pooled_index,axis=-1),index_new_nodes,old_shape)
        map_mask = tf.scatter_nd(ks.backend.expand_dims(pooled_index,axis=-1),tf.ones_like(index_new_nodes),old_shape)
        map_mask = tf.cast(map_mask,dtype=tf.bool)
        
        #Shift also edgeindex by batch offset
        shiftind = tf.cast(edgeind,dtype=index_dtype) #already shifted by batch offset (subgraphs)
        
        #Remove edges that were from filtered nodes via mask
        mask_edge = ks.backend.expand_dims(shiftind,axis=-1) == ks.backend.expand_dims(ks.backend.expand_dims(removed_index,axis=0),axis=0)  #this creates large tensor (batch*#edges,2,remove)
        mask_edge = tf.math.logical_not(ks.backend.any(ks.backend.any(mask_edge,axis=-1),axis=-1))
        clean_shiftind = shiftind[mask_edge]
        
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
        
        #Collect output tensors
        hidden_x = gated_n
        hidden_ix = pooled_id
        hidden_e = clean_edge_feat_sorted
        hidden_ie = out_indexlist
        pooled_nodes_mask = map_mask
        pooled_edges_mask = mask_edge
        
        
        out = [hidden_x, hidden_ix, hidden_e, hidden_ie, pooled_nodes_mask, pooled_edges_mask]
        return out
    
    
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