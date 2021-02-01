import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.ragged.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing
from kgcnn.layers.ragged.conv import DenseRagged
from kgcnn.layers.ragged.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingWeightedEdgesPerNode
from kgcnn.layers.ragged.set2set import Set2Set
from kgcnn.layers.ragged.casting import CastRaggedToDense,ChangeIndexing
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.ragged.mlp import MLPRagged


# 'Interaction Networks for Learning about Objects,Relations and Physics'
# by Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu
# http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
# https://github.com/higgsfield/interaction_network_pytorch


def getmodelINORP(  # Input
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
                    # Model specific parameter
                    is_sorted = True,
                    has_unconnected = False,
                    depth = 3,
                    node_dim = [100,50],
                    edge_dim = [100,100,100,100,50],
                    state_dim = [],
                    use_bias = True,
                    activation = 'relu',
                    use_set2set = False, #not in original paper
                    set2set_dim = 32, #not in original paper
                    pooling_method = "segment_mean",
                    **kwargs):
    """
    Generate Interaction network.
    
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
        
        is_sorted (bool, optional): Edge indices are sorted. Defaults to True.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to False.
        depth (int): Number of convolution layers. Default is 3.
        node_dim (list): Hidden node dimension for multiple kernels. Default is [100,50].
        edge_dim (list): Hidden edge dimension for multiple kernels. Default is [100,100,100,100,50].
        state_dim (list): Hidden state dimension. Default is [].
        use_bias (bool): Use bias for hidden conv units. Default is True.
        activation (str): Use activation for hidden conv units. Default is 'relu'.
        use_set2set (str): Use set2set pooling for graph embedding. Default is False.
        set2set_dim (int): Set2set dimension. Default is 32.
        pooling_method (str): Pooling method. Default is "segment_mean".
    
    Returns:
        model (tf.keras.model): Interaction model.
    
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
        
    #Preprocessing
    edi = ChangeIndexing()([n,edge_index_input])
     
    ev = GatherState()([uenv,n])
    # n-Layer Step
    for i in range(0,depth):
        #upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing(node_indexing = 'batch')([n,edi])
        eu2 = GatherNodesOutgoing(node_indexing = 'batch')([n,edi])
        upd = ks.layers.Concatenate(axis=-1)([eu2,eu1])
        eu = ks.layers.Concatenate(axis=-1)([upd,ed])
        
        for j in range(len(edge_dim)-1):
            eu = DenseRagged(edge_dim[j],use_bias=use_bias,activation=activation)(eu)
        eu = DenseRagged(edge_dim[-1],use_bias=use_bias,activation=activation)(eu)
        # Pool message
        nu = PoolingEdgesPerNode(pooling_method= pooling_method,is_sorted=is_sorted,has_unconnected=has_unconnected,node_indexing = 'batch')([n,eu,edi]) # Summing for each node connection
        # Add environment
        nu = ks.layers.Concatenate()([n,nu,ev]) # Concatenate node features with new edge updates
    
        for j in range(len(node_dim)-1):
            nu = DenseRagged(node_dim[j],use_bias=use_bias,activation=activation)(nu)
        n = DenseRagged(node_dim[-1],use_bias=use_bias,activation='linear')(nu)
    
    
    if(output_embedd == 'graph'):
        if(use_set2set == True):
            #output
            outSS = DenseRagged(set2set_dim)(n)
            out = Set2Set(set2set_dim)(outSS)
        else:
            out = PoolingNodes()(n)
        
        main_output = MLP(output_dim,
                        mlp_use_bias = output_use_bias,
                        mlp_activation = output_activation,
                        mlp_activity_regularizer=output_kernel_regularizer,
                        mlp_kernel_regularizer=output_kernel_regularizer,
                        mlp_bias_regularizer=output_bias_regularizer)(out)    
        
    else: #Node labeling
        out = n    
        main_output = MLPRagged(output_dim,
                                mlp_use_bias = output_use_bias,
                                mlp_activation = output_activation,
                                mlp_activity_regularizer=output_kernel_regularizer,
                                mlp_kernel_regularizer=output_kernel_regularizer,
                                mlp_bias_regularizer=output_bias_regularizer)(out)    
        
        
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm
    

    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input,env_input], outputs=main_output)
    
    return model

