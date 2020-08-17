# Implementation details

The major issue for graphs is their flexible size and shape, when using mini-batches. To handle flexible input tensors with keras,
either zero-padding plus masking or ragged/sparse tensors can be used. 
Depending on the task those representations can also be combined by casting from one to the other.
For more flexibility and a flatten batch-dimension, a dataloader from tf.keras.utils.Sequence is typically used. 

* Ragged Tensor:
Here the nodelist of shape (batch,None,nodefeatures) and edgelist of shape (batch,None,edgefeatures) are given by ragged tensors with ragged dimension (None,).
The graph structure is represented by an indexlist of shape (batch,None,2) with index of incoming i and outgoing j node as (i,j). 
The first index of incoming node i is expected to be sorted for faster pooling opertions. Furthermore the graph is directed, so an additional edge with (j,i) is required for undirected graphs.
In principle also the adjacency matrix can be represented as ragged tensor of shape (batch,None,None) but will be dense within each graph.

* Padded Tensor:
The node- and edgelists are given by a full-rank tensor of shape (batch,Nmax,features) with Nmax being the maximum number of edges or nodes in the dataset, 
by padding all unused entries with zero and marking them in an additional mask tensor of shape (batch,Nmax). 
This is only practical for highly connected graphs of similar shapes. 
Applying the mask is done by simply multiplying with the mask tensor. For pooling layers tf.boolean_mask() may be slower but can be favourable.
Besides the adjacencymatrix also the index list can be arranged in a matrix form with a max number of edges for faster node pooling, e.g. (batch,N,M) with number of nodes N and edges per Node M.


* Sparse Tensor:
...
