import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint,CastValuesToRagged
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.utils.activ import shifted_softplus
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.disjoint.conv import cfconv

# Model Schnet as defined 
# by Schuett et al. 2018
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566  
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


  
    
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
        self.lay_cfconv = cfconv(self.Alldim,activation=self.activation,use_bias=self.use_bias_cfconv,cfconv_pool = self.cfconv_pool,has_unconnected=self.has_unconnected,is_sorted=self.is_sorted)
        self.lay_dense1 = ks.layers.Dense(units=self.Alldim,activation='linear',use_bias =False)
        self.lay_dense2 = ks.layers.Dense(units=self.Alldim,activation=self.activation,use_bias =self.use_bias)
        self.lay_dense3 = ks.layers.Dense(units=self.Alldim,activation='linear',use_bias =self.use_bias)
        self.lay_add = ks.layers.Add()     
    def build(self, input_shape):
        """Build layer."""
        super(interaction, self).build(input_shape)
    def call(self, inputs):
        """Forward pass: Calculate node update.
            
        Args:
            [node,node_len,edge,edge_len,edge_index]
        
        Returns:
            node_update
        """
        node, bn, edge,edge_len, indexlis = inputs
        x = self.lay_dense1(node)
        x = self.lay_cfconv([x,bn,edge,edge_len,indexlis])
        x = self.lay_dense2(x)
        x = self.lay_dense3(x)
        out = self.lay_add([node ,x])
        return out




def getmodelSchnet(
                # Input
                input_node_shape,
                input_edge_shape,
                input_state_shape,
                input_node_vocab = 95,
                input_edge_vocab = 10,
                input_state_vocab = 100,
                input_node_embedd = 128,
                input_edge_embedd = 64,
                input_state_embedd = 64,
                input_type = 'ragged', 
                # Output
                output_embedd = 'graph',
                output_use_bias = [True,True,True],
                output_dim = [128,64,1],
                output_activation = ['shifted_softplus','shifted_softplus','linear'],
                output_kernel_regularizer = [None,None,None],
                output_activity_regularizer = [None,None,None],
                output_bias_regularizer = [None,None,None],
                output_type = 'padded',
                #Model specific
                depth = 4,
                node_dim = 128,
                use_bias = True,
                activation = 'shifted_softplus',
                cfconv_pool = "segment_sum",
                out_pooling_method = "segment_sum",
                out_scale_pos = 0,
                is_sorted= True,
                has_unconnected=False ,
                **kwargs
                ):
    """
    Make uncompiled Schnet model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_node_vocab (int): Node input embedding vocabulary. Default is 95.
        input_edge_vocab (int): Edge input embedding vocabulary. Default is 10.
        input_state_vocab (int): State input embedding vocabulary. Default is 100.
        input_node_embedd (int): Node input embedding dimension. Default is 128.
        input_edge_embedd (int): Edge input embedding dimension. Default is 64.
        input_state_embedd (int): State embedding dimension. Default is 64.
        input_type (str): Specify input type. Only 'ragged' is supported. 
        
        output_embedd (str): Graph or node embedding of the graph network. Default is 'graph'.
        output_use_bias (bool,list): Use bias for output. Optionally list for multiple layer. Defautl is [True,True,True].
        output_dim (list): Output dimension. Optionally list for multiple layer. Default is [32,16,1].
        output_activation (list): Activation function. Optionally list for multiple layer. Defautl is ['softplus2','softplus2','sigmoid'].
        output_kernel_regularizer (list): Kernel regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_activity_regularizer (list): Activity regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_bias_regularizer (list): Bias regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_type (str): Tensor output type. Default is 'padded'.
        
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
    if(len(input_node_shape) == 1):
        node_input = ks.layers.Input(shape=input_node_shape,name='node_input',dtype ="float32",ragged=True)
        n =  ks.layers.Embedding(input_node_vocab, input_node_embedd , name='node_embedding')(node_input)
    else:
        node_input = ks.layers.Input(shape=input_node_shape,name='node_input',dtype ="float32",ragged=True)
        n = node_input
        
    if(len(input_edge_shape)== 1):
        edge_input = ks.layers.Input(shape=input_edge_shape,name='edge_input',dtype ="float32",ragged=True)
        ed = ks.layers.Embedding(input_edge_vocab, input_state_embedd , name='edge_embedding')(edge_input)
    else:
        edge_input = ks.layers.Input(shape=input_edge_shape,name='edge_input',dtype ="float32",ragged=True)
        ed = edge_input
        
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    
    if(len(input_state_shape) == 0):
        env_input = ks.Input(shape=input_state_shape, dtype='float32' ,name='state_input')
        uenv = ks.layers.Embedding(input_state_vocab,input_state_embedd, name='state_embedding')(env_input)
    else:
        env_input = ks.Input(shape=input_state_shape, dtype='float32' ,name='state_input')
        uenv = env_input

    
    n,node_len,ed,edge_len,edi = CastRaggedToDisjoint()([n,ed,edge_index_input])
    
    if(isinstance(activation,str)):
        if(activation == 'shifted_softplus'):
            activation = shifted_softplus
    

    if(len(input_node_shape)>1 and input_node_shape[-1] != node_dim):
        n = ks.layers.Dense(node_dim,activation='linear')(n)
    
    
    for i in range(0,depth):
        n = interaction(node_dim,use_bias=use_bias,activation=activation,cfconv_pool=cfconv_pool)([n,node_len,ed,edge_len,edi])
     
    if(len(output_dim)>1):
        n = MLP(output_dim[:-1],
                mlp_use_bias = output_use_bias[:-1],
                mlp_activation = output_activation[:-1],
                mlp_activity_regularizer=output_kernel_regularizer[:-1],
                mlp_kernel_regularizer=output_kernel_regularizer[:-1],
                mlp_bias_regularizer=output_bias_regularizer[:-1])(n) 
    
    
    if(output_embedd == 'graph'):
        if(out_scale_pos == 0):
            n = ks.layers.Dense(output_dim[-1],activation=output_activation[-1],use_bias=output_use_bias[-1],bias_regularizer=output_bias_regularizer[-1],activity_regularizer=output_activity_regularizer[-1],kernel_regularizer=output_kernel_regularizer[-1])(n)
        out = PoolingNodes(pooling_method=out_pooling_method)([n,node_len])
        if(out_scale_pos == 1):
            out = ks.layers.Dense(output_dim[-1],activation=output_activation[-1],use_bias=output_use_bias[-1],bias_regularizer=output_bias_regularizer[-1],activity_regularizer=output_activity_regularizer[-1],kernel_regularizer=output_kernel_regularizer[-1])(out)
        main_output =  ks.layers.Flatten()(out) #will be dense
    else: #node embedding
        out = ks.layers.Dense(output_dim[-1],activation=output_activation[-1],use_bias=output_use_bias[-1],bias_regularizer=output_bias_regularizer[-1],activity_regularizer=output_activity_regularizer[-1],kernel_regularizer=output_kernel_regularizer[-1])(n)
        main_output = CastValuesToRagged()([out,node_len])
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm
    
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input,env_input], outputs=main_output)    

    return model