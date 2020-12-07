"""
Graph Convolutional Networks
from Kipf & Welling (ICLR 2017)
https://tkipf.github.io/graph-convolutional-networks/

@author: Patrick
"""

import numpy as np
import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.ragged.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing
from kgcnn.layers.ragged.conv import DenseRagged,ActivationRagged
from kgcnn.layers.ragged.pooling import PoolingWeightedEdgesPerNode,PoolingNodes


def precompute_adjacency_scaled(A,add_identity = True):
    """Precompute D^-0.5*(A+I)*D^-0.5"""
    A = np.array(A,dtype=np.float)
    if(add_identity==True):
        A = A +  np.identity(A.shape[0])
    d_ii = 1/np.sqrt(np.sum(A,axis=-1))
    D = np.zeros_like(A)
    D[np.arange(A.shape[0]),np.arange(A.shape[0])] = d_ii
    return np.matmul(D,np.matmul(A,D))

def scaled_adjacency_to_list(A,Ascaled):
    """Transfer adjacency to list"""
    A = np.array(A,dtype=np.bool)
    edge_weight = Ascaled[A]
    index1 = np.tile(np.expand_dims(np.arange(0,A.shape[0]),axis=1),(1,A.shape[1]))
    index2 = np.tile(np.expand_dims(np.arange(0,A.shape[1]),axis=0),(A.shape[0],1))
    index12 = np.concatenate([np.expand_dims(index1,axis=-1), np.expand_dims(index2,axis=-1)],axis=-1)
    edge_index = index12[A]
    return edge_index,edge_weight



def getmodelGCN(
            input_nodedim,
            input_type = "ragged",
            depth = 3,
            node_dim = [100,50,50],
            output_dim = [25,10,1],
            use_bias = False,
            activation = 'relu',
            use_pooling = True,
            output_activ = 'sigmoid',
            **kwargs
            ):


    node_input = ks.layers.Input(shape=(None,input_nodedim),name='node_input',dtype ="float32",ragged=True)
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    edge_input = ks.layers.Input(shape=(None,1),name='edge_input',dtype ="float32",ragged=True)
        
    n = node_input
    ed = edge_input
    edi = edge_index_input


    # n-Layer Step
    for i in range(0,depth):
        #upd = GatherNodes()([n,edi])
        eu = GatherNodesOutgoing()([n,edi])
        eu = DenseRagged(node_dim[i],use_bias=use_bias,activation='linear')(eu)
        nu = PoolingWeightedEdgesPerNode(pooling_method= 'segment_sum')([n,eu,edi,ed]) # Summing for each node connection
        n = ActivationRagged(activation=activation)(nu)

    if(use_pooling==True):
        out = PoolingNodes()(n)
    else:
        out = n    
    
    if(len(output_dim)>0):
        for j in range(len(output_dim)-1):
            out =  ks.layers.Dense(output_dim[j],activation=activation,use_bias=use_bias)(out)
        main_output =  ks.layers.Dense(output_dim[-1],name='main_output',activation=output_activ)(out)
    else:
        main_output = out
    
    model = ks.models.Model(inputs=[node_input,edge_index_input,edge_input], outputs=main_output)
    
    return model

