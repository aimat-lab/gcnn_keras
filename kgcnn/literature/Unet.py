import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint,CastValuesToRagged
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.layers.ragged.conv import DenseRagged
from kgcnn.layers.disjoint.topk import PoolingTopK,UnPoolingTopK
from kgcnn.layers.disjoint.connect import AdjacencyPower
from kgcnn.layers.disjoint.mlp import MLP


# Graph U-Nets
# by Hongyang Gao, Shuiwang Ji
# https://arxiv.org/pdf/1905.05178.pdf


def getmodelUnet(
                # Input
                input_node_shape,
                input_edge_shape,
                input_state_shape,
                input_node_vocab = 95,
                input_edge_vocab = 10,
                input_state_vocab = 100,
                input_node_embedd = 64,
                input_edge_embedd = 64,
                input_state_embedd = 64,
                input_type = 'ragged', 
                # Output
                output_embedd = 'graph',
                output_use_bias = [True,False],
                output_dim = [25,1],
                output_activation = ['relu','sigmoid'],
                output_kernel_regularizer = [None,None],
                output_activity_regularizer = [None,None],
                output_bias_regularizer = [None,None],
                output_type = 'padded',
                #Model specific
                hidden_dim = 32,
                depth = 4,
                k = 0.3,
                score_initializer = 'ones',
                use_bias = True,
                activation = 'relu',
                is_sorted=False,
                has_unconnected=True,
                use_reconnect = True
                ):
    """
    Make Graph U Net.
    
    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_node_vocab (int): Node input embedding vocabulary. Default is 95.
        input_edge_vocab (int): Edge input embedding vocabulary. Default is 10.
        input_state_vocab (int): State input embedding vocabulary. Default is 100.
        input_node_embedd (int): Node input embedding dimension. Default is 64.
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
        
        hidden_dim (int): Hidden node feature dimension 32,
        depth (int): Depth of pooling steps. Default is 4.
        k (float): Pooling ratio. Default is 0.3.
        score_initializer (str): How to initialize score kernel. Default is 'ones'.
        use_bias (bool): Use bias. Default is True.
        activation (str): Activation function used. Default is 'relu'.
        is_sorted (bool, optional): Edge indices are sorted. Defaults to True.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to False.
        use_reconnect (bool): Reconnect nodes after pooling. I.e. A=A^2. Default is True.
    
    Returns:
        model (ks.models.Model): Unet model.
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
    
    n,node_len,ed,edge_len,edi = CastRaggedToDisjoint()([n,ed,edge_index_input])
    
    
    in_graph = [n,node_len,ed,edge_len,edi]
    
    graph_list = [in_graph]
    map_list = []
    
    # U Down
    i_graph = in_graph
    for i in range(0,depth):
        
        n,node_len,ed,edge_len,edi = i_graph
        #GCN layer
        eu = GatherNodesOutgoing()([n,node_len,edi,edge_len])
        eu = ks.layers.Dense(hidden_dim,use_bias=use_bias,activation='linear')(eu)
        nu = PoolingEdgesPerNode(pooling_method= 'segment_mean',is_sorted=is_sorted,has_unconnected=has_unconnected)([n,node_len,eu,edge_len,edi]) # Summing for each node connection
        n = ks.layers.Activation(activation=activation)(nu)    
        
        if(use_reconnect == True):
            edi,ed,edge_len = AdjacencyPower(n=2)([edi,ed,edge_len,node_len])
   
        #Pooling
        i_graph,i_map = PoolingTopK(k=k,kernel_initializer=score_initializer)([n,node_len,ed,edge_len,edi])
        
        graph_list.append(i_graph)
        map_list.append(i_map)
    
    # U Up
    ui_graph = i_graph
    for i in range(depth,0,-1):
        o_graph = graph_list[i-1]
        i_map = map_list[i-1]
        ui_graph = UnPoolingTopK()(o_graph+i_map+ui_graph)
        
        n,node_len,ed,edge_len,edi = ui_graph
        #skip connection
        n = ks.layers.Add()([n,o_graph[0]])
        #GCN
        eu = GatherNodesOutgoing()([n,node_len,edi,edge_len])
        eu = ks.layers.Dense(hidden_dim,use_bias=use_bias,activation='linear')(eu)
        nu = PoolingEdgesPerNode(pooling_method= 'segment_mean',is_sorted=is_sorted,has_unconnected=has_unconnected)([n,node_len,eu,edge_len,edi]) # Summing for each node connection
        n = ks.layers.Activation(activation=activation)(nu)
        
        ui_graph = [n,node_len,ed,edge_len,edi]

    
    #Otuput
    n = ui_graph[0]
    node_len = ui_graph[1]
    if(output_embedd == 'graph'):
        out = PoolingNodes(pooling_method='segment_mean')([n,node_len])
        
        out = MLP(output_dim,
                mlp_use_bias = output_use_bias,
                mlp_activation = output_activation,
                mlp_activity_regularizer=output_kernel_regularizer,
                mlp_kernel_regularizer=output_kernel_regularizer,
                mlp_bias_regularizer=output_bias_regularizer)(out)     
        main_output = ks.layers.Flatten()(out) #will be dense
    else: #node embedding
        out = MLP(output_dim,
                mlp_use_bias = output_use_bias,
                mlp_activation = output_activation,
                mlp_activity_regularizer=output_kernel_regularizer,
                mlp_kernel_regularizer=output_kernel_regularizer,
                mlp_bias_regularizer=output_bias_regularizer)(n)     
        main_output = CastValuesToRagged()([out,node_len])
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input,env_input], outputs=main_output)
        
    return model    
        
        