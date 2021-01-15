import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.conv import DenseDisjoint
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint


# Neural Message Passing for Quantum Chemistry
# by 
# http://arxiv.org/abs/1704.01212    


class ApplyMessage(ks.layers.Layer):
    """
    Message Pass.
    
    Args:
        shape_msg (int): Message dimension.
    Inputs:
        [edge_message, nodes]
    Outputs
        edge updates
    """
    def __init__(self,shape_msg,**kwargs):
        """Initialize layer."""
        super(ApplyMessage, self).__init__(**kwargs) 
        self.mat_shape_msg = shape_msg
    def build(self, input_shape):
        """Build layer."""
        super(ApplyMessage, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        dens_e,dens_n = inputs
        dens_m= tf.reshape(dens_e,(ks.backend.shape(dens_e)[0],self.mat_shape_msg,self.mat_shape_msg))
        out = tf.keras.backend.batch_dot(dens_m,dens_n) 
        return out     
    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        shape_msg,shape_node = input_shape
        return (shape_node[0],self.mat_shape_msg)


class GRUupdate(ks.layers.Layer):
    """
    Gated recurrent unit update.
    
    Args:
        channels (int):
    Inputs:
        [nodes, updates]
    Outputs
        node updates    
    """
    def __init__(self,channels,**kwargs):
        """Initialize layer."""
        super(GRUupdate, self).__init__(**kwargs) 
        self.gru = tf.keras.layers.GRUCell(channels)
    def build(self, input_shape):
        """Build layer."""
        #self.gru.build(channels)
        super(GRUupdate, self).build(input_shape)          
    def call(self, inputs):
        """Forward pass."""
        n,eu = inputs
        # Apply GRU for update node state
        out,_ = self.gru(eu,[n])
        return out     
    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        shape_node,shape_msg = input_shape
        return (shape_node[0],self.channels)


def getmodelNMPN(
            input_nodedim,
            input_edgedim,
            input_envdim,
            output_dim,
            nvocal: int = 95,
            embedding_dim: int = 16,
            input_type = "ragged",
            depth = 3,
            node_dim = 128,
            use_set2set = True,
            set2set_dim = 32,
            use_bias = True,
            activation = 'selu',
            is_sorted:bool = True,
            has_unconnected:bool = False,
            **kwargs
            ):
    """
    Get Message passing model.

    Args:
        input_nodedim (int): Input node dimension.
        input_edgedim (int): Edge size.
        input_envdim (int): Dimension of environment.
        output_dim (int): Output dimension.
        nvocal (int): Vocabulary for emebdding layer. Default is 95.
        embedding_dim (int): Dimension for emebdding layer. Default is 16.
        input_type (str, optional): Input tensor type, only ragged supported. Defaults to "ragged".
        depth (int, optional): Depth. Defaults to 3.
        node_dim (int, optional): Dimension for hidden node representation. Defaults to 128.
        use_set2set (bool, optional): Use set2set layer. Defaults to True.
        set2set_dim (bool, optional): Dimension for set2set. Defaults to 32.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str, optional): Activation function to use. Defaults to 'selu'.
        is_sorted (bool): Are edge indices sorted. Default is True.
        has_unconnected (bool): Has unconnected nodes. Default is False.
        **kwargs

    Returns:
        model (ks.models.Model): Message Passing model.

    """

    if(input_nodedim == None):
        node_input = ks.layers.Input(shape=(None,),name='node_input',dtype ="float32",ragged=True)
        n =  ks.layers.Embedding(nvocal, embedding_dim, name='node_embedding')(node_input)
    else:
        node_input = ks.layers.Input(shape=(None,input_nodedim),name='node_input',dtype ="float32",ragged=True)
        n = node_input
    edge_input = ks.layers.Input(shape=(None,input_edgedim),name='edge_input',dtype ="float32",ragged=True)
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    #env_input = ks.Input(shape=(input_envdim,), dtype='float32' ,name='state_feature_input')
    
    n,node_len,ed,edge_len,edi = CastRaggedToDisjoint()([n,edge_input,edge_index_input])
    #uenv = env_input
       
    n = DenseDisjoint(node_dim)(n)
    EdgNet = DenseDisjoint(node_dim*node_dim,activation=activation)(ed)
    gru = GRUupdate(node_dim)
    

    for i in range(0,depth):
        eu = GatherNodesOutgoing()([n,edi])
        eu = ApplyMessage(node_dim)([EdgNet,eu])
        eu = PoolingEdgesPerNode(is_sorted=is_sorted,has_unconnected=has_unconnected)([n,eu,edi]) # Summing for each node connections
        n = gru([n,eu])
    
    if(use_set2set == True):
        #output
        outSS = DenseDisjoint(set2set_dim)(n)
        out = Set2Set(set2set_dim)([outSS,node_len])
    else:
        out = PoolingNodes()([n,node_len])
        
    main_output =  ks.layers.Dense(output_dim,name='main_output')(out)
    
   
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input], outputs=main_output)

    return model