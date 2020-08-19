"""@package: Model Schnet as defined by Schuett et al. 2018
https://doi.org/10.1063/1.5019779
https://arxiv.org/abs/1706.08566  
https://aip.scitation.org/doi/pdf/10.1063/1.5019779

In Schnetpack there are more refined output modules for e.g. stress, dipole and atmoic values.
This is not adapted here. Only standard network as in paper above.
"""

import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.conv import ConvFlatten
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.batch import RaggedToDisjoint,CastListToRagged,CastRaggedToList,CorrectIndexListForSubGraph


def ssp(x):
    """
    Args:
        x : single values to apply activation to using tf.keras functions
    Returns:
        out : log(exp(x)+1) - log(2)
    """
    return ks.activations.softplus(x) - ks.backend.log(2.0)


class cfconvList(ks.layers.Layer):
    def __init__(self, Alldim=64, activation='selu',use_bias = True,cfconv_pool = tf.math.segment_sum,**kwargs):
        super(cfconvList, self).__init__(**kwargs)
        self.activation = activation
        self.use_bias = use_bias
        self.cfconv_pool=cfconv_pool
        self.Alldim = Alldim
        self.lay_dense1 = ConvFlatten(channels=self.Alldim,activation=self.activation,use_bias=self.use_bias)
        self.lay_dense2 = ConvFlatten(channels=self.Alldim,activation='linear',use_bias=self.use_bias)
        self.lay_sum = PoolingEdgesPerNode(pooling_method=self.cfconv_pool)
        self.gather_n = GatherNodesOutgoing()
    def build(self, input_shape):
        super(cfconvList, self).build(input_shape)
    def call(self, inputs):
        #Calculate edge Update
        node, edge, indexlis, bn = inputs
        x = self.lay_dense1(edge)
        x = self.lay_dense2(x)
        node2Exp = self.gather_n([node,indexlis])
        x = node2Exp*x
        x= self.lay_sum([x,indexlis])
        return x
    def compute_output_shape(self, input_shape):
        return (input_shape)
    
class interaction(ks.layers.Layer):
    def __init__(self, Alldim=64, activation='selu',use_bias_cfconv=True,use_bias = True,cfconv_pool=tf.math.segment_sum ,**kwargs):
        super(interaction, self).__init__(**kwargs)
        self.activation = activation
        self.use_bias = use_bias
        self.use_bias_cfconv = use_bias_cfconv
        self.cfconv_pool = cfconv_pool
        self.Alldim = Alldim
        self.lay_cfconv = cfconvList(self.Alldim,activation=self.activation,use_bias=use_bias_cfconv,cfconv_pool = self.cfconv_pool)    
        self.lay_dense1 = ConvFlatten(channels=self.Alldim,activation='linear',use_bias =False)
        self.lay_dense2 = ConvFlatten(channels=self.Alldim,activation=self.activation,use_bias =self.use_bias)
        self.lay_dense3 = ConvFlatten(channels=self.Alldim,activation='linear',use_bias =self.use_bias)
        self.lay_add = ks.layers.Add()     
    def build(self, input_shape):
        super(interaction, self).build(input_shape)
    def call(self, inputs):
        #Calculate edge Update
        node, edge, indexlis, bn = inputs
        x = self.lay_dense1(node)
        x = self.lay_cfconv([x,edge,indexlis,bn])
        x = self.lay_dense2(x)
        x = self.lay_dense3(x)
        out = self.lay_add([node ,x])
        return out
    def compute_output_shape(self, input_shape):
        return (input_shape)




def getmodelSchnet(
            input_nodedim,
            input_edgedim,
            output_dim,
            input_type = "ragged",
            Depth = 4,
            nvocal: int = 95,
            node_dim = 128,
            use_bias = True,
            activation = ssp,
            cfconv_pool = "sum",
            out_MLP = [128,64],
            use_pooling=True,
            out_pooling_method = "sum",
            out_scale_pos = 0,
            **kwargs
            ):


    if input_nodedim is None:
        node_input =  ks.Input(shape=(None,),dtype='int32', name='atom_int_input',ragged=True)  # only z as feature
        n =  ks.layers.Embedding(nvocal, node_dim, name='atom_embedding')(node_input)
    else:
        node_input = ks.Input(shape=(None,input_nodedim), dtype='float32' ,name='atom_feature_input',ragged=True)
        n = node_input
    edge_input = ks.layers.Input(shape=(None,input_edgedim),name='edge_input',dtype ="float32",ragged=True)
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    
    n,node_len,ed,edge_len,edi = RaggedToDisjoint()([n,edge_input,edge_index_input])
    
    
    if(input_nodedim is not None and input_nodedim != node_dim):
        n = ConvFlatten(node_dim,activation='linear')(n)
    
    
    for i in range(0,Depth):
        n = interaction(node_dim,use_bias=use_bias,activation=activation,cfconv_pool=cfconv_pool)([n,ed,edi,node_len])
     
    if(len(out_MLP)>0):
        for i in range(len(out_MLP)-1):
            n = ConvFlatten(out_MLP[i],activation=activation,use_bias=use_bias)(n)
        n = ConvFlatten(out_MLP[-1],activation='linear',use_bias=use_bias)(n)
        
    if(use_pooling==True):
        if(out_scale_pos == 0):
            n = ConvFlatten(output_dim,activation='linear',use_bias=use_bias)(n)
        out = PoolingNodes(pooling_method=out_pooling_method)([n,node_len])
        if(out_scale_pos == 1):
            out = ks.layers.Dense(output_dim,activation='linear')(out)
        main_output =  ks.layers.Flatten(name='main_output')(out)
    else:
        main_output = n
    
    

    if(use_pooling==False):
        main_output = CastListToRagged()([main_output,node_len])
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input], outputs=main_output)    

    return model