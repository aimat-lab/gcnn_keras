import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint,CastValuesToRagged
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.disjoint.update import ApplyMessage,GRUupdate
from kgcnn.layers.ragged.conv import DenseRagged
from kgcnn.layers.ragged.casting import CastRaggedToDense

# Neural Message Passing for Quantum Chemistry
# by Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
# http://arxiv.org/abs/1704.01212    



def getmodelNMPN(
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
                output_activation = ['selu','selu','sigmoid'],
                output_kernel_regularizer = [None,None,None],
                output_activity_regularizer = [None,None,None],
                output_bias_regularizer = [None,None,None],
                output_type = 'padded',
                #Model specific
                depth = 3,
                node_dim = 128,
                use_set2set = True,
                set2set_dim = 32,
                use_bias = True,
                activation = 'selu',
                is_sorted:bool = True,
                has_unconnected:bool = False,
                set2set_init:str = '0',
                set2set_pool:str = "sum",
                out_pool = "segment_sum",
                **kwargs
            ):
    """
    Get Message passing model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_node_vocab (int): Node input embedding vocabulary. Default is 95.
        input_edge_vocab (int): Edge input embedding vocabulary. Default is 5.
        input_state_vocab (int): State input embedding vocabulary. Default is 100.
        input_node_embedd (int): Node input embedding dimension. Default is 64.
        input_edge_embedd (int): Edge input embedding dimension. Default is 64.
        input_state_embedd (int): State embedding dimension. Default is 4.
        input_type (str): Specify input type. Only 'ragged' is supported. 
        
        output_embedd (str): Graph or node embedding of the graph network. Default is 'graph'.
        output_use_bias (bool,list): Use bias for output. Optionally list for multiple layer. Defautl is [True,True,True].
        output_dim (list): Output dimension. Optionally list for multiple layer. Default is [25,10,1].
        output_activation (list): Activation function. Optionally list for multiple layer. Defautl is ['selu','selu','sigmoid'].
        output_kernel_regularizer (list): Kernel regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_activity_regularizer (list): Activity regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_bias_regularizer (list): Bias regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_type (str): Tensor output type. Default is 'padded'.
        
        depth (int, optional): Depth. Defaults to 3.
        node_dim (int, optional): Dimension for hidden node representation. Defaults to 128.
        use_set2set (bool, optional): Use set2set layer. Defaults to True.
        set2set_dim (bool, optional): Dimension for set2set. Defaults to 32.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str, optional): Activation function to use. Defaults to 'selu'.
        is_sorted (bool): Are edge indices sorted. Default is True.
        has_unconnected (bool): Has unconnected nodes. Default is False.
        set2set_init (str): Initialize method. Default is '0'.
        set2set_pool (str): Pooling method in set2set. Default is "sum".
        out_pool (str): Final node pooling in place of set2set.
        
        **kwargs

    Returns:
        model (ks.models.Model): Message Passing model.
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
    #uenv = env_input
       
    n = ks.layers.Dense(node_dim)(n)
    EdgNet = ks.layers.Dense(node_dim*node_dim,activation=activation)(ed)
    gru = GRUupdate(node_dim)
    

    for i in range(0,depth):
        eu = GatherNodesOutgoing()([n,node_len,edi,edge_len])
        eu = ApplyMessage(node_dim)([EdgNet,eu])
        eu = PoolingEdgesPerNode(is_sorted=is_sorted,has_unconnected=has_unconnected)([n,node_len,eu,edge_len,edi]) # Summing for each node connections
        n = gru([n,eu])
      
    
    if(output_embedd == 'graph'):
        if(use_set2set == True):
            #output
            outSS = ks.layers.Dense(set2set_dim)(n)
            out = Set2Set(set2set_dim,pooling_method=set2set_pool,init_qstar = set2set_init)([outSS,node_len])
        else:
            out = PoolingNodes(pooling_method = out_pool)([n,node_len])
        
        # final dense layers 
        main_output = MLP(output_dim,
                        mlp_use_bias = output_use_bias,
                        mlp_activation = output_activation,
                        mlp_activity_regularizer=output_kernel_regularizer,
                        mlp_kernel_regularizer=output_kernel_regularizer,
                        mlp_bias_regularizer=output_bias_regularizer)(out)         
        
    else: #Node labeling
        out = n    
        for j in range(len(output_dim)-1):
            out =  DenseRagged(output_dim[j],activation=output_activation[j],use_bias=output_use_bias[j],
                               bias_regularizer=output_bias_regularizer[j],activity_regularizer=output_activity_regularizer[j],kernel_regularizer=output_kernel_regularizer[j])(out)
        main_output = DenseRagged(output_dim[-1],name='main_output',activation=output_activation[-1],use_bias=output_use_bias[-1],
                                  bias_regularizer=output_bias_regularizer[-1],activity_regularizer=output_activity_regularizer[-1],kernel_regularizer=output_kernel_regularizer[-1])(out)
        
        main_output = CastValuesToRagged()([main_output,node_len])
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm
    

    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input,env_input], outputs=main_output)

    return model