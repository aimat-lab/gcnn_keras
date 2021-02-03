import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from kgcnn.layers.ragged.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingWeightedEdgesPerNode
from kgcnn.layers.ragged.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing
from kgcnn.utils.activ import kgcnn_custom_act
    


class DenseRagged(tf.keras.layers.Layer):
    """
    Custom Dense Layer for ragged input. The dense layer can be used as convolutional unit.
    
    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        
    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
        
    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """
    
    def __init__(self, 
                  units,
                  activation=None,
                  use_bias=True,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  **kwargs):
        """Initialize layer same as Dense."""
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        
        super(DenseRagged, self).__init__(**kwargs)
                
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = ks.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.bias_initializer = ks.initializers.get(bias_initializer)
        self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
        self.bias_regularizer = ks.regularizers.get(bias_regularizer)
        self.kernel_constraint = ks.constraints.get(kernel_constraint)
        self.bias_constraint = ks.constraints.get(bias_constraint)
    
        self._supports_ragged_inputs = True 
        
    def build(self, input_shape):
        """Build layer's kernel and bias."""
        last_dim = input_shape[-1]
        
        # Add Kernel 
        self.kernel = self.add_weight( 'kernel',
                                        shape=[last_dim, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        # Add bias
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
    
        super(DenseRagged, self).build(input_shape)  #should set sef.built = True
        
    def call(self, inputs):
        """Forward pass."""
        outputs = tf.ragged.map_flat_values(tf.matmul,inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add,outputs, self.bias)
        
        outputs =  tf.ragged.map_flat_values(self.activation,outputs)
        return outputs
    
    def get_config(self):
        """Update config."""
        config = super(DenseRagged, self).get_config()
        config.update({
        'units':
            self.units,
        'activation':
            ks.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            ks.initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            ks.initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            ks.regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            ks.regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            ks.constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            ks.constraints.serialize(self.bias_constraint)
        })
        return config
 
    
class ActivationRagged(tf.keras.layers.Layer):
    """
    Applies an activation function to an output.
    
    Arguments:
        activation: Activation function, such as `tf.nn.relu`, or string name of built-in.
    """
    
    def __init__(self, activation, **kwargs):
        """Initialize layer same as Activation."""
        super(ActivationRagged, self).__init__(**kwargs)
        self.activation = ks.activations.get(activation)
        self._supports_ragged_inputs = True 
    def call(self, inputs):
        """Forward pass.
        
        Args:
            tensor (tf.ragged): Ragged tensor of shape e.g. (batch,None,F)
        
        Returns:
            tf.ragged: Elementwise activation of flat values.
        """
        out = tf.ragged.map_flat_values(self.activation,inputs)
        return out

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape

    def get_config(self):
        """Update config."""
        config = {'activation': ks.activations.serialize(self.activation)}
        base_config = super(ActivationRagged, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    


class GCN(ks.layers.Layer):
    """
    Graph convolution according to Kipf et al.
    
    Computes graph conv as $sigma(A*(WX+b))$ where A is the precomputed adjacency matrix.
    In place of A, edges and edge indices are used. A is considered pre-sacled. Otherwise use e.g. segment-mean, scale by weights etc.
    Edges must be broadcasted to node feautres X.
    
    Args:
        units (int): Output dimension/ units of dense layer.
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        activation (str): Activation function 'relu'.
        pooling_method (str): Pooling method for summing edges 'segment_sum'.
        use_bias (bool): Whether to use bias. Default is False,
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        **kwargs
    """
    
    def __init__(self, 
                 units,
                 node_indexing = 'batch',
                 activation='relu',
                 pooling_method= 'segment_sum',
                 use_bias = False,
                 is_sorted=False,
                 has_unconnected=True,
                 normalize_by_weights = False,
                 **kwargs):
        """Initialize layer."""
        super(GCN, self).__init__(**kwargs)
        self.units = units
        self.node_indexing = node_indexing 
        self.normalize_by_weights = normalize_by_weights
        self.use_bias = use_bias
        self.pooling_method = pooling_method
        self.has_unconnected = has_unconnected
        self.is_sorted = is_sorted
        self.activation = activation
        
        self.deserial_activation = ks.activations.deserialize(activation,custom_objects=kgcnn_custom_act) if isinstance(activation,str) or isinstance(activation,dict) else activation
        #Layers
        self.lay_gather = GatherNodesOutgoing(node_indexing = self.node_indexing)
        self.lay_dense = DenseRagged(self.units,use_bias=self.use_bias,activation='linear')
        self.lay_pool = PoolingWeightedEdgesPerNode(pooling_method= self.pooling_method,is_sorted=self.is_sorted,
                                                    has_unconnected=self.has_unconnected,node_indexing = self.node_indexing,
                                                    normalize_by_weights = self.normalize_by_weights)
        self.lay_act = ActivationRagged(self.deserial_activation)
        self._supports_ragged_inputs = True  
    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass.
        
        Inputs list of [node, edge, edge_index]
        
        Args: 
            nodes (tf.ragged): Ragged node feature list of shape (batch,None,F)
            edges (tf.ragged): Ragged edge feature list of shape (batch,None,F)
            edge_index (tf.ragged): Edge indices for (batch,None,2) 
        
        Returns:
            features (tf.ragged): A list of updated node features.        
            Output shape is (batch,None,F).
        """
        node,edges,edge_index = inputs
        no = self.lay_gather([node,edge_index])
        no = self.lay_dense(no)
        nu = self.lay_pool([node,no,edge_index,edges]) # Summing for each node connection
        out = self.lay_act(nu)
        return out     
    def get_config(self):
        """Update config."""
        config = super(GatherNodesOutgoing, self).get_config()
        config.update({"units": self.units})
        config.update({"node_indexing": self.node_indexing})
        config.update({"normalize_by_weights": self.normalize_by_weights})
        config.update({"use_bias": self.use_bias})
        config.update({"pooling_method": self.pooling_method})
        config.update({"has_unconnected": self.has_unconnected})
        config.update({"is_sorted": self.is_sorted})
        config.update({"activation": self.activation})
        return config 


