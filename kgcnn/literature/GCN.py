import numpy as np
import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.ragged.casting import CastRaggedToDense,ChangeIndexing
from kgcnn.layers.ragged.conv import GCN,DenseRagged
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.layers.ragged.mlp import MLPRagged
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.disjoint.pooling import PoolingNodes

# 'Semi-Supervised Classification with Graph Convolutional Networks'
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn
       


def getmodelGCN(
                    # Input
                    input_node_shape,
                    input_edge_shape,
                    input_state_shape,
                    input_node_vocab = 100,
                    input_edge_vocab = 10,
                    input_state_vocab = 100,
                    input_node_embedd = 64,
                    input_edge_embedd = 64,
                    input_state_embedd = 64,
                    input_type = 'ragged', 
                    # Output
                    output_embedd = 'graph',
                    output_use_bias = [True,True,False],
                    output_dim = [25,10,1],
                    output_activation = ['relu','relu','sigmoid'],
                    output_kernel_regularizer = [None,None,None],
                    output_activity_regularizer = [None,None,None],
                    output_bias_regularizer = [None,None,None],
                    output_type = 'padded',
                    #Model specific
                    depth = 3,
                    hidden_dim = 100, 
                    use_bias = False,
                    activation = 'relu',
                    graph_labeling = False,
                    is_sorted=True,
                    has_unconnected=False ,
                    **kwargs):
    """
    Make GCN model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_node_vocab (int): Node input embedding vocabulary. Default is 100.
        input_edge_vocab (int): Edge input embedding vocabulary. Default is 10.
        input_state_vocab (int): State input embedding vocabulary. Default is 100.
        input_node_embedd (int): Node input embedding dimension. Default is 64.
        input_edge_embedd (int): Edge input embedding dimension. Default is 64.
        input_state_embedd (int): State embedding dimension. Default is 64.
        input_type (str): Specify input type. Only 'ragged' is supported. 
        
        output_embedd (str): Graph or node embedding of the graph network. Default is 'graph'.
        output_use_bias (bool,list): Use bias for output. Optionally list for multiple layer. Defautl is [True,True,False].
        output_dim (list): Output dimension. Optionally list for multiple layer. Default is [25,10,1].
        output_activation (list): Activation function. Optionally list for multiple layer. Defautl is ['relu','relu','sigmoid'].
        output_kernel_regularizer (list): Kernel regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_activity_regularizer (list): Activity regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_bias_regularizer (list): Bias regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_type (str): Tensor output type. Default is 'padded'.
        
        depth (int, optional): Number of convolutions. Defaults to 3.
        hidden_dim (int, optional): Hidden node dimension W(i). Defaults to 100.
        use_bias (bool, optional): To use bias. Defaults to False.
        activation (str, optional): Activation after convolution. Defaults to 'relu'.
        graph_labeling (bool, optional): Pooling nodes for graph embedding. Defaults to False.
        is_sorted (bool, optional): Edge indices are sorted for first index. Defaults to True.
        has_unconnected (boool, optional): Graph has isolated nodes. Defaults to False.
        **kwargs

    Returns:
        model (tf.keras.models.Model): uncompiled model.

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
        
        
    n = DenseRagged(hidden_dim,use_bias=use_bias,activation='linear')(n)
    ed = ed
    edi = ChangeIndexing()([n,edge_index_input])


    # n-Layer Step
    for i in range(0,depth):
        n = GCN(hidden_dim,use_bias=use_bias,activation=activation,pooling_method= 'segment_sum',is_sorted=is_sorted,has_unconnected=has_unconnected,node_indexing = 'batch')([n,ed,edi])
        

    if(graph_labeling==True):
        out = PoolingNodes()(n) #will return tensor
        out = MLP(output_dim,
                    mlp_use_bias = output_use_bias,
                    mlp_activation = output_activation,
                    mlp_activity_regularizer=output_kernel_regularizer,
                    mlp_kernel_regularizer=output_kernel_regularizer,
                    mlp_bias_regularizer=output_bias_regularizer)(out)  
    
    else: #Node labeling
        out = n    
        out = MLPRagged(output_dim,
                    mlp_use_bias = output_use_bias,
                    mlp_activation = output_activation,
                    mlp_activity_regularizer=output_kernel_regularizer,
                    mlp_kernel_regularizer=output_kernel_regularizer,
                    mlp_bias_regularizer=output_bias_regularizer)(out)  
        out = CastRaggedToDense()(out) # no ragged for distribution supported atm
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input], outputs=out)
    
    return model

