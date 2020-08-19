"""@package: Neural Message Passing for Quantum Chemistry
http://arxiv.org/abs/1704.01212    

"""
import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.conv import ConvFlatten
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.batch import RaggedToDisjoint,CastListToRagged,CastRaggedToList,CorrectIndexListForSubGraph



class ApplyMessage(ks.layers.Layer):
    def __init__(self,shape_msg,**kwargs):
        super(ApplyMessage, self).__init__(**kwargs) 
        self.mat_shape_msg = shape_msg
    def build(self, input_shape):
        super(ApplyMessage, self).build(input_shape)          
    def call(self, inputs):
        dens_e,dens_n = inputs
        dens_m= tf.reshape(dens_e,(ks.backend.shape(dens_e)[0],self.mat_shape_msg,self.mat_shape_msg))
        out = tf.keras.backend.batch_dot(dens_m,dens_n) 
        return out     
    def compute_output_shape(self, input_shape):
        shape_msg,shape_node = input_shape
        return (shape_node[0],self.mat_shape_msg)


class GRUupdate(ks.layers.Layer):
    def __init__(self,channels,**kwargs):
        super(GRUupdate, self).__init__(**kwargs) 
        self.gru = tf.keras.layers.GRUCell(channels)
    def build(self, input_shape):
        #self.gru.build(channels)
        super(GRUupdate, self).build(input_shape)          
    def call(self, inputs):
        n,eu = inputs
        # Apply GRU for update node state
        out,_ = self.gru(eu,[n])
        return out     
    def compute_output_shape(self, input_shape):
        shape_node,shape_msg = input_shape
        return (shape_node[0],self.channels)


def getmodelNMPN(
            input_nodedim,
            input_edgedim,
            input_envdim,
            output_dim,
            input_type = "ragged",
            Depth = 3,
            node_dim = 128,
            use_set2set = True,
            set2set_dim = 32,
            use_bias = True,
            activation = 'selu',
            **kwargs
            ):

        

    node_input = ks.layers.Input(shape=(None,input_nodedim),name='node_input',dtype ="float32",ragged=True)
    edge_input = ks.layers.Input(shape=(None,input_edgedim),name='edge_input',dtype ="float32",ragged=True)
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    env_input = ks.Input(shape=(input_envdim,), dtype='float32' ,name='state_feature_input')
    
    n,node_len,ed,edge_len,edi = RaggedToDisjoint()([node_input,edge_input,edge_index_input])
    uenv = env_input
       
        
    
    n = ConvFlatten(node_dim)(n)
    EdgNet = ConvFlatten(node_dim*node_dim,activation=activation)(ed)
    gru = GRUupdate(node_dim)
    

    for i in range(0,Depth):
        eu = GatherNodesOutgoing()([n,edi])
        eu = ApplyMessage(node_dim)([EdgNet,eu])
        eu = PoolingEdgesPerNode()([eu,edi]) # Summing for each node connections
        n = gru([n,eu])
    
    if(use_set2set == True):
        #output
        outSS = ConvFlatten(set2set_dim)(n)
        out = Set2Set(set2set_dim)([outSS,node_len])
    else:
        out = PoolingNodes()([n,node_len])
        
    main_output =  ks.layers.Dense(output_dim,name='main_output')(out)
    
   
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input], outputs=main_output)

    return model