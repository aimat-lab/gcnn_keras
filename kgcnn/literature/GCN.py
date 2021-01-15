import numpy as np
import tensorflow.keras as ks
import tensorflow as tf
import scipy.sparse as sp

from kgcnn.layers.ragged.gather import GatherNodesOutgoing,SampleToBatchIndexing
from kgcnn.layers.ragged.conv import DenseRagged,ActivationRagged
from kgcnn.layers.ragged.pooling import PoolingWeightedEdgesPerNode,PoolingNodes
from kgcnn.layers.ragged.casting import CastRaggedToDense


# 'Semi-Supervised Classification with Graph Convolutional Networks'
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn


def precompute_adjacency_scaled(A,add_identity = True):
    """
    Precompute the scaled adjacency matrix A_scaled = D^-0.5 (A + I) D^-0.5.

    Args:
        A (np.array,scipy.sparse): Adjacency matrix of shape (N,N).
        add_identity (bool, optional): Whether to add identity. Defaults to True.

    Returns:
        np.array: D^-0.5 (A + I) D^-0.5.

    """
    if(isinstance(A,np.ndarray)):
        A = np.array(A,dtype=np.float)
        if(add_identity==True):
            A = A +  np.identity(A.shape[0])
        rowsum = np.sum(A,axis=-1)
        d_ii = np.power(rowsum, -0.5).flatten()
        d_ii[np.isinf(d_ii)] = 0.
        D = np.zeros_like(A)
        D[np.arange(A.shape[0]),np.arange(A.shape[0])] = d_ii
        return np.matmul(D,np.matmul(A,D))
    elif(isinstance(A,sp.csr.csr_matrix) or
         isinstance(A,sp.coo.coo_matrix)):
        adj = sp.coo_matrix(A)
        adj = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj.sum(1))
        d_ii = np.power(rowsum, -0.5).flatten()
        d_ii[np.isinf(d_ii)] = 0.
        D = sp.diags(d_ii)
        return adj.transpose().dot(D).transpose().dot(D).tocoo()    
    else:
        raise TypeError("Error: Matrix format not supported:",type(A))


def scaled_adjacency_to_list(A,Ascaled):
    """
    Map adjacency matrix to index list plus edge weights.

    Args:
        A (np.array): Original Adjacency matrix of shape (N,N).
        Ascaled (np.array,scipy.sparse): Scaled Adjacency matrix of shape (N,N). A_scaled = D^-0.5 (A + I) D^-0.5.

    Returns:
        edge_index (np.array): Indexlist of shape (N,2).
        edge_weight (np.array): Entries of Adjacency matrix of shape (N,N)
        
    """
    if(isinstance(A,np.ndarray)):
        A = np.array(A,dtype=np.bool)
        edge_weight = Ascaled[A]
        index1 = np.tile(np.expand_dims(np.arange(0,A.shape[0]),axis=1),(1,A.shape[1]))
        index2 = np.tile(np.expand_dims(np.arange(0,A.shape[1]),axis=0),(A.shape[0],1))
        index12 = np.concatenate([np.expand_dims(index1,axis=-1), np.expand_dims(index2,axis=-1)],axis=-1)
        edge_index = index12[A]
        return edge_index,edge_weight
    elif(isinstance(Ascaled,sp.csr.csr_matrix) or
         isinstance(Ascaled,sp.coo.coo_matrix)):
        ei1 =  np.array(Ascaled.row.tolist(),dtype=np.int)
        ei2 =  np.array(Ascaled.col.tolist(),dtype=np.int)
        edge_index = np.concatenate([np.expand_dims(ei1,axis=-1),np.expand_dims(ei2,axis=-1)],axis=-1)
        edge_weight = np.array(Ascaled.data)
        return edge_index,edge_weight
    else:
        raise TypeError("Error: Matrix format not supported:",type(A))
        


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

