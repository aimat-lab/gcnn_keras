import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K



   
class GatherNodes(ks.layers.Layer):
    """
    Gather nodes from ragged tensor by indices provided by a ragged index tensor in mini-batches.
    
    An edge at index is the connection for node(index([0])) to node(index([1]))
    The features of gathered ingoing and outgoing nodes are concatenated according to index tensor.
    
    Args:
        ragged_validate (bool): False
        node_indexing (str): 'sample'
        **kwargs
        
    Example:
        out = GatherNodes()([input_node,input_edge_index])   
    """
    
    def __init__(self, 
                 ragged_validate = False,
                 node_indexing = 'sample',
                 **kwargs):
        """Initialize layer."""
        super(GatherNodes, self).__init__(**kwargs) 
        self.ragged_validate = ragged_validate
        self.node_indexing = node_indexing
        self._supports_ragged_inputs = True          
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodes, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [nodes, edge_index] 
        
        Args:
            nodes (tf.ragged): Node feature tensor of shape (batch,None,F)
            edge_index (tf.ragged): Ragged edge indices of shape (batch,None,2)
        
        Returns:
            features (tf.ragged): Gathered node features with entries at index (node(index([0])),node(index([1]))) of shape (batch,None,F+F)
            The length matches the index Tensor.
        """
        nod,edgeind = inputs
        if(self.node_indexing == 'batch'):
            shiftind = edgeind.values
        elif(self.node_indexing == 'sample'):
            shift1 = edgeind.values
            shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        dens = nod.values
        g1 = tf.gather(dens,shiftind[:,0])
        g2 = tf.gather(dens,shiftind[:,1])
        get = tf.concat([g1,g2],axis=1)
        out = tf.RaggedTensor.from_row_splits(get,edgeind.row_splits,validate=self.ragged_validate)         
        #out = edgeind.with_values(get)
        return out     
    def get_config(self):
        """Update config."""
        config = super(GatherNodes, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"node_indexing": self.node_indexing})
        return config 


class GatherNodesOutgoing(ks.layers.Layer):
    """
    Gathers Outgoing Nodes from ragged tensor by index provided by a ragged index tensor in mini-batches.
    
    An edge at index is the connection for node(index([0])) to node(index([1]))
    The feature of gathered outgoing nodes are the connected nodes at index[1].
    
    Args:
        ragged_validate (bool): False
        node_indexing (str): 'sample'
        **kwargs
        
    Example:
        out = GatherNodesOutgoing()([input_node,input_edge_index])   
    """
    
    def __init__(self, 
                 ragged_validate = False,
                 node_indexing = 'sample',
                 **kwargs):
        """Initialize layer."""
        super(GatherNodesOutgoing, self).__init__(**kwargs) 
        self.ragged_validate = ragged_validate
        self.node_indexing = node_indexing
        self._supports_ragged_inputs = True          
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesOutgoing, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [nodes, edge_index]
        
        Args:
            nodes (tf.ragged): Node feature tensor of shape (batch,None,F)
            edge_index (tf.ragged): Ragged edge indices of shape (batch,None,2)
            
        Returns:
            features (tf.ragged): Gathered outgoing nodes with entries at index 
            node(index([1])) of shape (batch,None,F)
            The length matches the index Tensor at axis=1.
        """
        nod,edgeind = inputs
        if(self.node_indexing == 'batch'):
            shiftind = edgeind.values
        elif(self.node_indexing == 'sample'): 
            shift1 = edgeind.values
            shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype) 
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        nodind = shiftind
        dens = nod.values
        g2= tf.gather(dens,nodind[:,1])
        out = tf.RaggedTensor.from_row_splits(g2,edgeind.row_splits,validate=self.ragged_validate)         
        #out = edgeind.with_values(g2)
        return out  
    def get_config(self):
        """Update config."""
        config = super(GatherNodesOutgoing, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"node_indexing": self.node_indexing})
        return config 
  
    
class GatherNodesIngoing(ks.layers.Layer):
    """
    Gathers ingoing nodes from ragged tensor by index provided by a ragged index tensor in mini-batches.
    
    An edge at index is the connection for node(index([0])) to node(index([1]))
    The feature of gathered ingoing nodes at index[0] for the edges in edge tensor.
    
    Args:
        ragged_validate (bool): False
        node_indexing (str): 'sample'
        **kwargs
        
    Example:
        out = GatherNodesIngoing()([input_node,input_edge_index])   
    """
    
    def __init__(self,
                 ragged_validate = False,
                 node_indexing = 'sample',
                 **kwargs):
        """Initialize layer."""
        super(GatherNodesIngoing, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True
        self.node_indexing = node_indexing
        self.ragged_validate = ragged_validate        
    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesIngoing, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs of List [nodes, edge_index]
        
        Args:
            nodes (tf.ragged): Node feature tensor of shape (batch,None,F)
            edge_index (tf.ragged): Ragged edge indices of shape (batch,None,2)
            
        Returns:
            features (tf.ragged): Gathered ingoing nodes with entries at index 
            node(index([1])) of shape (batch,None,F)
            The length matches the index Tensor at axis=1.
        """
        nod,edgeind = inputs
        if(self.node_indexing == 'batch'):
            shiftind = edgeind.values
        elif(self.node_indexing == 'sample'): 
            shift1 = edgeind.values
            shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1],edgeind.row_lengths()),axis=1)
            shiftind = shift1 + tf.cast(shift2,dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")
        nodind = shiftind
        dens = nod.values
        g1= tf.gather(dens,nodind[:,0])
        out = tf.RaggedTensor.from_row_splits(g1,edgeind.row_splits,validate=self.ragged_validate)         
        #out = edgeind.with_values(g1)
        return out  
    def get_config(self):
        """Update config."""
        config = super(GatherNodesIngoing, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"node_indexing": self.node_indexing})
        return config 

    
class GatherState(ks.layers.Layer):
    """
    Gathers a global state for nodes or edges.
    
    Args:
        ragged_validate (bool): False
        **kwargs
        
    Example:
        out = GatherState()([state,nodes])   
    """
    
    def __init__(self,
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True
        self.ragged_validate = ragged_validate          
    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs List of [state, target] 
        
        Args:
            state (tf.tensor): Environment or global graph state tensor of shape (batch,F)
            target (tf.raged): Ragged node/edgelist (batch,None,F)
            
        Returns:
            states (tf.ragged): A ragged tensor with shape (batch,None,F)
            The corresponding state of each graph is repeated to 
            match the target tensor.
        """
        env,nod= inputs
        target_len = nod.row_lengths()
        out = tf.repeat(env,target_len,axis=0)
        out = tf.RaggedTensor.from_row_splits(out,nod.row_splits,validate=self.ragged_validate)
        return out           
    def get_config(self):
        """Update config."""
        config = super(GatherState, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        return config 

    
class LazyConcatenateNodes(ks.layers.Layer):
    """
    Concatenate ragged nodetensors without checking shape.
    
    Ragged dimension only at first axis. Can be replaced with standard concat function.
    
    Args:
        axis (int): Axis to concatenate. Default is -1.
        ragged_validate (bool): False
        **kwargs
    """

    def __init__(self,
                 axis = -1,
                 ragged_validate = False,
                 **kwargs):
        """Initialize layer."""
        super(LazyConcatenateNodes, self).__init__(**kwargs) 
        self._supports_ragged_inputs = True 
        self.ragged_validate = ragged_validate   
        self.axis = axis
    def build(self, input_shape):
        """Build layer."""
        super(LazyConcatenateNodes, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Args:
            nodes (list): [nodes,nodes,...] of shape [(batch,None,F),(batch,None,F),...]
            of ragged tensors of nodes with similar ragged dimension None. 
        
        Returns:
            nodes (tf.ragged): Concatenated Nodes with shape (batch,None,Sum(F)) 
            where the row_splits of first nodelist are kept.
        """
        out = tf.keras.backend.concatenate([x.values for x in inputs],axis=self.axis)
        out = tf.RaggedTensor.from_row_splits(out,inputs[0].row_splits,validate=self.ragged_validate) 
        return out     
    def get_config(self):
        """Update config."""
        config = super(LazyConcatenateNodes, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"axis": self.axis})
        return config 
