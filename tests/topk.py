"""
@author: Patrick
"""

import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

from kgcnn.layers.ragged.topk import PoolingTopK

def to_graph(it_n,it_ei,it_e,plotting=True):
    G=nx.Graph()
    for i in range(len(it_n)):
        G.add_node(i, nfeat = it_n[i])
    for i in range(len(it_ei)):
        G.add_edge(it_ei[i][0],it_ei[i][1], efeat= it_e[i])
    
    if(plotting==True):
        glabels = nx.get_node_attributes(G, 'nfeat')    
        nx.draw(G,node_size=100,labels=glabels)
        #plt.savefig(grname + ".pdf")
        plt.show()
    return G

n1 = [[[1.0], [6.0], [1.0], [6.0], [1.0], [1.0], [6.0], [6.0]], [[6.0], [1.0], [1.0], [1.0], [7.0], [1.0], [6.0], [8.0], [6.0], [1.0], [6.0], [7.0], [1.0], [1.0], [1.0]]]
ei1 = [[[0, 1], [1, 0], [1, 6], [2, 3], [3, 2], [3, 5], [3, 7], [4, 7], [5, 3], [6, 1], [6, 7], [7, 3], [7, 4], [7, 6]], [[0, 6], [0, 8], [0, 9], [1, 11], [2, 4], [3, 4], [4, 2], [4, 3], [4, 6], [5, 10], [6, 0], [6, 4], [6, 14], [7, 8], [8, 0], [8, 7], [8, 11], [9, 0], [10, 5], [10, 11], [10, 12], [10, 13], [11, 1], [11, 8], [11, 10], [12, 10], [13, 10], [14, 6]]]
e1 = [[[0.408248290463863], [0.408248290463863], [0.3333333333333334], [0.35355339059327373], [0.35355339059327373], [0.35355339059327373], [0.25], [0.35355339059327373], [0.35355339059327373], [0.3333333333333334], [0.2886751345948129], [0.25], [0.35355339059327373], [0.2886751345948129]], [[0.25], [0.25], [0.35355339059327373], [0.35355339059327373], [0.35355339059327373], [0.35355339059327373], [0.35355339059327373], [0.35355339059327373], [0.25], [0.3162277660168379], [0.25], [0.25], [0.35355339059327373], [0.35355339059327373], [0.25], [0.35355339059327373], [0.25], [0.35355339059327373], [0.3162277660168379], [0.22360679774997896], [0.3162277660168379], [0.3162277660168379], [0.35355339059327373], [0.25], [0.22360679774997896], [0.3162277660168379], [0.3162277660168379], [0.35355339059327373]]]

to_graph(n1[0],ei1[0],e1[0])
to_graph(n1[1],ei1[1],e1[1])

node = tf.ragged.constant(n1,ragged_rank=1,inner_shape=(1,))
edgeind =  tf.ragged.constant(ei1,ragged_rank=1,inner_shape=(2,))
edgefeat =  tf.ragged.constant(e1,ragged_rank=1,inner_shape=(1,))


out = PoolingTopK(k=0.3,kernel_initializer="ones",ragged_validate=True)([node,edgeind,edgefeat])

n2 = out[0].numpy()
ei2 = out[1].numpy()
e2 = out[2].numpy()
to_graph(n2[0],ei2[0],e2[0])
to_graph(n2[1],ei2[1],e2[1])

from kgcnn.layers.disjoint.topk import PoolingTopK as PoolingTopK2
from kgcnn.layers.disjoint.batch import RaggedToDisjoint

dislist = RaggedToDisjoint()([node,edgefeat,edgeind])

to_graph(dislist[0].numpy(),dislist[-1].numpy(),dislist[-3].numpy())

pool_dislist = PoolingTopK2(k=0.3,kernel_initializer="ones")(dislist)

to_graph(pool_dislist[0].numpy(),pool_dislist[-1].numpy(),pool_dislist[-3].numpy())