import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.mpnn_conv import GRUUpdate, TrafoEdgeNetMessages, MatMulMessages
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.pool.set2set import PoolingSet2Set
from kgcnn.utils.models import generate_embedding, update_model_kwargs
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer

# Neural Message Passing for Quantum Chemistry
# by Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
# http://arxiv.org/abs/1704.01212    

model_default = {'name': "NMPN",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                     "edge": {"input_dim": 5, "output_dim": 64}},
                 'output_embedding': 'graph',
                 'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                "activation": ['selu', 'selu', 'sigmoid']},
                 'gauss_args': {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
                 'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "sum",
                                  "init_qstar": "0"},
                 'pooling_args': {'pooling_method': "segment_sum"},
                 'edge_mlp': {'use_bias': True, 'activation': 'swish', "units": [64, 64, 64]},
                 'use_set2set': True, 'depth': 3, 'node_dim': 64,
                 "geometric_edge": False, "make_distance": False, "expand_distance": False,
                 'verbose': 1
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               output_embedding=None,
               output_mlp=None,
               gauss_args=None,
               set2set_args=None,
               pooling_args=None,
               edge_mlp=None,
               use_set2set=None,
               node_dim=None,
               depth=None,
               geometric_edge=None,
               make_distance=None,
               expand_distance=None,
               verbose=None,
               name=None):
    """Make NMPN graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.
        gauss_args (dict): Dictionary of layer arguments unpacked in `GaussBasisLayer` layer.
        set2set_args (dict): Dictionary of layer arguments unpacked in `PoolingSet2Set` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in `PoolingNodes`, `PoolingLocalEdges` layers.
        edge_mlp (dict): Dictionary of layer arguments unpacked in `MLP` layer for edge matrix.
        use_set2set (bool): Whether to use `PoolingSet2Set` layer.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of graph embedding units or depth of the network.
        geometric_edge (bool): Whether the edges are geometric, like distance or coordinates.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices indices. Expansion uses `gauss_args`.
        verbose (int): Level of verbosity.
        name (str): Name of the model.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])  # Or coordinates
    edge_index_input = ks.layers.Input(**inputs[2])
    edi = edge_index_input

    # embedding, if no feature dimension
    n0 = generate_embedding(node_input, inputs[0]['shape'], input_embedding['node'])
    if not geometric_edge:
        ed = generate_embedding(edge_input, inputs[1]['shape'], input_embedding['edge'])

    # If coordinates are in place of edges
    if make_distance:
        pos1, pos2 = NodePosition()([ed, edi])
        ed = NodeDistanceEuclidean()([pos1, pos2])

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Make hidden dimension
    n = DenseEmbedding(node_dim, activation="linear")(n0)

    # Make edge networks.
    edge_net_in = GraphMLP(**edge_mlp)(ed)
    edge_net_in = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_in)
    edge_net_out = GraphMLP(**edge_mlp)(ed)
    edge_net_out = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_out)

    # Gru for node updates
    gru = GRUUpdate(node_dim)

    for i in range(0, depth):
        n_in = GatherNodesOutgoing()([n, edi])
        n_out = GatherNodesIngoing()([n, edi])
        m_in = MatMulMessages()([edge_net_in, n_in])
        m_out = MatMulMessages()([edge_net_out, n_out])
        eu = LazyConcatenate(axis=-1)([m_in, m_out])
        eu = PoolingLocalEdges(**pooling_args)([n, eu, edi])  # Summing for each node connections
        n = gru([n, eu])

    n = LazyConcatenate(axis=-1)([n0, n])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            out = DenseEmbedding(set2set_args['channels'], activation="linear")(n)
            out = PoolingSet2Set(**set2set_args)(out)
        else:
            out = PoolingNodes(**pooling_args)(n)
        # final dense layers
        main_output = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        main_output = GraphMLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(main_output)
        # no ragged for distribution supported atm
    else:
        raise ValueError("Unsupported graph embedding for mode `NMPN`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model
