"""@package: Interaction Networks for Learning about Objects,Relations and Physics
http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
https://github.com/higgsfield/interaction_network_pytorch
"""
import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing
from kgcnn.layers.conv import ConvFlatten
from kgcnn.layers.pooling import PoolingEdgesPerNode,PoolingNodes
from kgcnn.layers.set2set import Set2Set
from kgcnn.layers.batch import RaggedToDisjoint,CastListToRagged,CastRaggedToList,CorrectIndexListForSubGraph


def getmodelINORP(
            input_nodedim,
            input_edgedim,
            input_envdim,
            input_type = "ragged",
            Depth = 1,
            edge_dim = [100,100,100,100,50],
            node_dim = [100,50],
            output_dim = [25,10,1],
            output_activ = 'sigmoid',
            use_bias = True,
            activation = 'relu',
            use_set2set = False, #not in original paper
            set2set_dim = 32,
            use_pooling = True,
            add_env = True,
            pooling_method = 'segment_mean',
            **kwargs
            ):


    node_input = ks.layers.Input(shape=(None,input_nodedim),name='node_input',dtype ="float32",ragged=True)
    edge_input = ks.layers.Input(shape=(None,input_edgedim),name='edge_input',dtype ="float32",ragged=True)
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    env_input = ks.Input(shape=(input_envdim,), dtype='float32' ,name='state_feature_input')
        
    n, node_len, ed, edge_len, edi = RaggedToDisjoint()([node_input,edge_input,edge_index_input])
    uenv = env_input
    
    
    ev = GatherState()([uenv,node_len])
    # n-Layer Step
    for i in range(0,Depth):
        #upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing()([n,edi])
        eu2 = GatherNodesOutgoing()([n,edi])
        upd = ks.layers.Concatenate(axis=-1)([eu2,eu1])
        eu = ks.layers.Concatenate(axis=-1)([upd,ed])
        for j in range(len(edge_dim)-1):
            eu = ConvFlatten(edge_dim[j],use_bias=use_bias,activation=activation)(eu)
        eu = ConvFlatten(edge_dim[-1],use_bias=use_bias,activation=activation)(eu)
        nu = PoolingEdgesPerNode(pooling_method= pooling_method )([eu,edi]) # Summing for each node connection
        if(add_env == True):
            nu = ks.layers.Concatenate()([n,nu,ev]) # Concatenate node features with new edge updates
        else:
            nu = ks.layers.Concatenate()([n,nu]) # Concatenate node features with new edge updates
        for j in range(len(node_dim)-1):
            nu = ConvFlatten(node_dim[j],use_bias=use_bias,activation=activation)(nu)
        n = ConvFlatten(node_dim[-1],use_bias=use_bias,activation='linear')(nu)
    
    if(use_set2set == True):
        #output
        outSS = ConvFlatten(set2set_dim)(n)
        out = Set2Set(set2set_dim)([outSS,node_len])
    elif(use_pooling==True):
        out = PoolingNodes()([n,node_len])
    else:
        out = n    
    
    if(len(output_dim)>0):
        for j in range(len(output_dim)-1):
            out =  ks.layers.Dense(output_dim[j],activation=activation,use_bias=use_bias)(out)
        main_output =  ks.layers.Dense(output_dim[-1],name='main_output',activation=output_activ)(out)
    else:
        main_output = out
    
    if(use_pooling==False):
        main_output = CastListToRagged()([main_output,node_len])
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input,env_input], outputs=main_output)
    
    return model

