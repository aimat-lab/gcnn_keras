import numpy as np
import tensorflow.keras as ks
import tensorflow as tf


from kgcnn.layers.ragged.gather import GatherNodesOutgoing,SampleToBatchIndexing
from kgcnn.layers.ragged.conv import DenseRagged,ActivationRagged
from kgcnn.layers.ragged.pooling import PoolingWeightedEdgesPerNode,PoolingNodes
from kgcnn.layers.ragged.casting import CastRaggedToDense


# 'Semi-Supervised Classification with Graph Convolutional Networks'
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn

       


def getmodelGCN(input_nodedim,
            input_type = "ragged",  #not used atm
            depth = 3,
            node_dim = 100, #input features to nodes dimension
            hidden_dim = 100, 
            output_dim = [100,50,1],
            use_bias = False,
            activation = 'relu',
            graph_labeling = False,
            output_activ = 'sigmoid',
            is_sorted=True,
            has_unconnected=False ,
            **kwargs):
    """
    Make GCN model.

    Args:
        input_nodedim (int): Node input dimension.
        input_type (int, optional): Input tensor. Not used atm. Defaults to "ragged".
        depth (int, optional): Number of convolutions. Defaults to 3.
        node_dim (int, optional): Node dimension W(0). Defaults to 100.
        hidden_dim (int, optional): Hidden node dimension W(i). Defaults to 100.
        output_dim (list, optional): Dimension at output. Defaults to [100,50,1].
        use_bias (bool, optional): To use bias. Defaults to False.
        activation (str, optional): Activation after convolution. Defaults to 'relu'.
        graph_labeling (bool, optional): Pooling nodes for graph embedding. Defaults to False.
        output_activ (str, optional): Last activation. Defaults to 'sigmoid'.
        is_sorted (bool, optional): Edge indices are sorted for first index. Defaults to True.
        has_unconnected (boool, optional): Graph has isolated nodes. Defaults to False.
        **kwargs

    Returns:
        model (tf.keras.models.Model): uncompiled model.

    """

    node_input = ks.layers.Input(shape=(None,input_nodedim),name='node_input',dtype ="float32",ragged=True)
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    edge_input = ks.layers.Input(shape=(None,1),name='edge_input',dtype ="float32",ragged=True)
        
    n = DenseRagged(node_dim,use_bias=use_bias,activation='linear')(node_input)
    ed = edge_input
    edi = SampleToBatchIndexing()([n,edge_index_input])


    # n-Layer Step
    for i in range(0,depth):
        #upd = GatherNodes()([n,edi])
        eu = GatherNodesOutgoing(node_indexing = 'batch')([n,edi])
        eu = DenseRagged(hidden_dim,use_bias=use_bias,activation='linear')(eu)
        nu = PoolingWeightedEdgesPerNode(pooling_method= 'segment_sum',is_sorted=is_sorted,has_unconnected=has_unconnected,node_indexing = 'batch')([n,eu,edi,ed]) # Summing for each node connection
        n = ActivationRagged(activation=activation)(nu)

    if(graph_labeling==True):
        out = PoolingNodes()(n) #will return tensor
        for j in range(len(output_dim)-1):
            out =  ks.layers.Dense(output_dim[j],activation=activation,use_bias=use_bias)(out)
        out =  ks.layers.Dense(output_dim[-1],name='main_output',activation=output_activ)(out)
    
    else: #Node labeling
        out = n    
        for j in range(len(output_dim)-1):
            out =  DenseRagged(output_dim[j],activation=activation,use_bias=use_bias)(out)
        out =  DenseRagged(output_dim[-1],name='main_output',activation=output_activ)(out)
        out = CastRaggedToDense()(out) # no ragged for distribution supported atm
    
    model = ks.models.Model(inputs=[node_input,edge_index_input,edge_input], outputs=out)
    
    return model

