import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.mpnn_conv import GRUUpdate, TrafoMatMulMessages
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pool.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.pool.set2set import PoolingSet2Set
from kgcnn.utils.models import generate_embedding, update_model_kwargs

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
                 'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "sum",
                                  "init_qstar": "0"},
                 'pooling_args': {'pooling_method': "segment_mean"},
                 'edge_dense': {'use_bias': True, 'activation': 'selu'},
                 'use_set2set': True, 'depth': 3, 'node_dim': 128,
                 'verbose': 1
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               output_embedding=None,
               output_mlp=None,
               set2set_args=None,
               pooling_args=None,
               edge_dense=None,
               use_set2set=None,
               node_dim=None,
               depth=None, **kwargs):
    """Make NMPN graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.
        set2set_args (dict): Dictionary of layer arguments unpacked in `PoolingSet2Set` layer.
        pooling_args (dict): Dictionary of layer arguments unpacked in `PoolingNodes`, `PoolingLocalEdges` layers.
        edge_dense (dict): Dictionary of layer arguments unpacked in `Dense` layer for edge matrix.
        use_set2set (bool): Whether to use `PoolingSet2Set` layer.
        node_dim (int): Dimension of hidden node embedding.
        depth (int): Number of graph embedding units or depth of the network.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # embedding, if no feature dimension
    n = generate_embedding(node_input, inputs[0]['shape'], input_embedding['node'])
    ed = generate_embedding(edge_input, inputs[1]['shape'], input_embedding['edge'])
    edi = edge_index_input

    # Model
    n = Dense(node_dim, activation="linear")(n)
    edge_net = Dense(node_dim * node_dim, **edge_dense)(ed)
    gru = GRUUpdate(node_dim)

    for i in range(0, depth):
        eu = GatherNodesOutgoing()([n, edi])
        eu = TrafoMatMulMessages(node_dim, )([edge_net, eu])
        eu = PoolingLocalEdges(**pooling_args)(
            [n, eu, edi])  # Summing for each node connections
        n = gru([n, eu])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            outss = Dense(set2set_args['channels'], activation="linear")(n)
            out = PoolingSet2Set(**set2set_args)(outss)
        else:
            out = PoolingNodes(**pooling_args)(n)

        # final dense layers
        main_output = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        main_output = MLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(main_output)
        # no ragged for distribution supported atm
    else:
        raise ValueError("Unsupported graph embedding for mode `NMPN`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model
