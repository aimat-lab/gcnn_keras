import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.conv import DenseDisjoint
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint,CastValuesToRagged
from kgcnn.layers.ragged.casting import CastRaggedToDense


# Model Schnet as defined 
# by Schuett et al. 2018
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566  
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


def shifted_softplus(x):
    """
    Shifted softplus activation function.
    
    Args:
        x : single values to apply activation to using tf.keras functions
    Returns:
        out : log(exp(x)+1) - log(2)
    """
    return ks.activations.softplus(x) - ks.backend.log(2.0)


class cfconvList(ks.layers.Layer):
    """
    Convolution layer.
    
    Args:
        Alldim (int): 64 
        activation (str): 'selu'
        use_bias (bool): True
        cfconv_pool (str): 'segment_sum'
        is_sorted (bool): True
        has_unconnected (bool): False
    
    Input:
        [node,edge,edge_index,node_len]
    
    Output:
        node_update
    """
    
    def __init__(self, Alldim=64, 
                 activation='selu',
                 use_bias = True,
                 cfconv_pool = 'segment_sum',
                 is_sorted = True,
                 has_unconnected=False,
                 **kwargs):
        """Initialize Layer."""
        super(cfconvList, self).__init__(**kwargs)
        self.activation = activation
        self.use_bias = use_bias
        self.cfconv_pool=cfconv_pool
        self.Alldim = Alldim
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        #Layer
        self.lay_dense1 = DenseDisjoint(units=self.Alldim,activation=self.activation,use_bias=self.use_bias)
        self.lay_dense2 = DenseDisjoint(units=self.Alldim,activation='linear',use_bias=self.use_bias)
        self.lay_sum = PoolingEdgesPerNode(pooling_method=self.cfconv_pool,is_sorted = self.is_sorted , has_unconnected=self.has_unconnected)
        self.gather_n = GatherNodesOutgoing()
    def build(self, input_shape):
        """Build layer."""
        super(cfconvList, self).build(input_shape)
    def call(self, inputs):
        """Forward pass: Calculate edge update."""
        node, edge, indexlis, bn = inputs
        x = self.lay_dense1(edge)
        x = self.lay_dense2(x)
        node2Exp = self.gather_n([node,indexlis])
        x = node2Exp*x
        x= self.lay_sum([node,x,indexlis])
        return x
    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return (input_shape)
 
    
    
class interaction(ks.layers.Layer):
    """
    Interaction block.
    
    Args:
        Alldim (int): 64 
        activation (str): 'selu'
        use_bias (bool): True
        cfconv_pool (str): 'segment_sum'
        is_sorted (bool): True
        has_unconnected (bool): False
    
    Input:
        [node,edge,edge_index,node_len]
    
    Output:
        node_update
    """
    
    def __init__(self, Alldim=64, 
                 activation='selu',
                 use_bias_cfconv=True,
                 use_bias = True,
                 cfconv_pool='segment_sum',
                 is_sorted = True,
                 has_unconnected=False,
                 **kwargs):
        """Initialize Layer."""
        super(interaction, self).__init__(**kwargs)
        self.activation = activation
        self.use_bias = use_bias
        self.use_bias_cfconv = use_bias_cfconv
        self.cfconv_pool = cfconv_pool
        self.Alldim = Alldim
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        #Layers
        self.lay_cfconv = cfconvList(self.Alldim,activation=self.activation,use_bias=self.use_bias_cfconv,cfconv_pool = self.cfconv_pool,has_unconnected=self.has_unconnected,is_sorted=self.is_sorted)
        self.lay_dense1 = DenseDisjoint(units=self.Alldim,activation='linear',use_bias =False)
        self.lay_dense2 = DenseDisjoint(units=self.Alldim,activation=self.activation,use_bias =self.use_bias)
        self.lay_dense3 = DenseDisjoint(units=self.Alldim,activation='linear',use_bias =self.use_bias)
        self.lay_add = ks.layers.Add()     
    def build(self, input_shape):
        """Build layer."""
        super(interaction, self).build(input_shape)
    def call(self, inputs):
        """Forward pass: Calculate node update."""
        node, edge, indexlis, bn = inputs
        x = self.lay_dense1(node)
        x = self.lay_cfconv([x,edge,indexlis,bn])
        x = self.lay_dense2(x)
        x = self.lay_dense3(x)
        out = self.lay_add([node ,x])
        return out
    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return (input_shape)




def getmodelSchnet(
            input_nodedim,
            input_edgedim,
            output_dim,
            input_type = "ragged",
            depth = 4,
            nvocal: int = 95,
            node_dim = 128,
            use_bias = True,
            activation = shifted_softplus,
            cfconv_pool = "segment_sum",
            out_MLP = [128,64], #MLP at the end
            graph_embedd = True,
            out_pooling_method = "segment_sum",
            out_scale_pos = 0,
            output_activation = 'linear',
            is_sorted= True,
            has_unconnected=False ,
            **kwargs
            ):
    """
    Make uncompiled Schnet model.

    Args:
        input_nodedim (int): Input node dim.
        input_edgedim (int): Input edge dim.
        output_dim (int): Label dimension.
        input_type (str, optional): Input type, only ragged. Defaults to "ragged".
        depth (int, optional): Number of interaction units. Defaults to 4.
        nvocal (int, optional): Embedding vocabulary. Defaults to 95.
        node_dim (int, optional): Hidden node dim. Defaults to 128.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str, optional): Activation function. Defaults to shifted_softplus.
        cfconv_pool (str, optional): Pooling method. Defaults to "segment_sum".
        out_MLP (list, optional): Layer dimension of final MLP. Defaults to [128,64].
        graph_embedd (bool, optional): Graph or node embedding. Defaults to True.
        out_pooling_method (str, optional): Node pooling method. Defaults to "segment_sum".
        out_scale_pos (int, optional): Scaling output, position of layer. Defaults to 0.
        output_activation (str, optional): Last activation. Defaults to 'linear'.
        is_sorted (bool, optional): Edge indices are sorted. Defaults to True.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to False.
        **kwargs

    Returns:
        model (tf.keras.models.Model): Schnet.

    """
    if input_nodedim is None:
        node_input =  ks.Input(shape=(None,),dtype='int32', name='atom_int_input',ragged=True)  # only z as feature
        n =  ks.layers.Embedding(nvocal, node_dim, name='atom_embedding')(node_input)
    else:
        node_input = ks.Input(shape=(None,input_nodedim), dtype='float32' ,name='atom_feature_input',ragged=True)
        n = node_input
    edge_input = ks.layers.Input(shape=(None,input_edgedim),name='edge_input',dtype ="float32",ragged=True)
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    
    n,node_len,ed,edge_len,edi = CastRaggedToDisjoint()([n,edge_input,edge_index_input])
    
    
    if(input_nodedim is not None and input_nodedim != node_dim):
        n = DenseDisjoint(node_dim,activation='linear')(n)
    
    
    for i in range(0,depth):
        n = interaction(node_dim,use_bias=use_bias,activation=activation,cfconv_pool=cfconv_pool)([n,ed,edi,node_len])
     
    if(len(out_MLP)>0):
        for i in range(len(out_MLP)-1):
            n = DenseDisjoint(out_MLP[i],activation=activation,use_bias=use_bias)(n)
        n = DenseDisjoint(out_MLP[-1],activation='linear',use_bias=use_bias)(n)
    
    
    if(graph_embedd==True):
        if(out_scale_pos == 0):
            n = DenseDisjoint(output_dim,activation=output_activation,use_bias=use_bias)(n)
        out = PoolingNodes(pooling_method=out_pooling_method)([n,node_len])
        if(out_scale_pos == 1):
            out = ks.layers.Dense(output_dim,activation=output_activation)(out)
        main_output =  ks.layers.Flatten(name='main_output')(out) #will be dense
    else: #node embedding
        out = DenseDisjoint(output_dim,activation=output_activation)(n)
        main_output = CastValuesToRagged()([out,node_len])
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input], outputs=main_output)    

    return model