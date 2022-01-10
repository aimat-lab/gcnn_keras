import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate, ActivationEmbedding, LazyAdd, DropoutEmbedding, \
    OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.conv.dmpnn_conv import DMPNNPPoolingEdgesDirected
from kgcnn.utils.models import update_model_kwargs

# Analyzing Learned Molecular Representations for Property Prediction
# by Kevin Yang, Kyle Swanson, Wengong Jin, Connor Coley, Philipp Eiden, Hua Gao,
# Angel Guzman-Perez, Timothy Hopper, Brian Kelley, Miriam Mathea, Andrew Palmer,
# Volker Settels, Tommi Jaakkola, Klavs Jensen, and Regina Barzilay
# https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237

model_default = {'name': "DMPNN",
                 'inputs': [
                     {'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                     {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                     {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
                     {'shape': (None, 1), 'name': "edge_indices_reverse", 'dtype': 'int64', 'ragged': True}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                     "edge": {"input_dim": 5, "output_dim": 64}},
                 'pooling_args': {'pooling_method': "sum"},
                 "edge_initialize": {"units": 128, 'use_bias': True, 'activation': 'relu'},
                 'edge_dense': {"units": 128, 'use_bias': True, 'activation': 'linear'},
                 "edge_activation": {"activation": "relu"},
                 "node_dense": {"units": 128, 'use_bias': True, 'activation': 'relu'},
                 'verbose': 10, "depth": 5, "dropout": {"rate": 0.1},
                 'output_embedding': 'graph',
                 'output_mlp': {"use_bias": [True, True, False], "units": [64, 32, 1],
                                "activation": ['relu', 'relu', 'linear']}
                 }


@update_model_kwargs(model_default)
def make_model(name=None,
               inputs=None,
               input_embedding=None,
               pooling_args=None,
               edge_initialize=None,
               edge_dense=None,
               edge_activation=None,
               node_dense=None,
               dropout=None,
               depth=None,
               verbose=None,
               output_embedding=None,
               output_mlp=None
               ):
    """Make DMPNN graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        name (str): Name of the model. Should be "DMPNN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        pooling_args (dict): Dictionary of layer arguments unpacked in `PoolingNodes`, `PoolingLocalEdges` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in `Dense` layer for first edge embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in `Dense` layer for edge embedding.
        edge_activation (dict): Edge Activation after skip connection.
        node_dense (dict): Dense kwargs for node embedding layer.
        depth (int): Number of graph embedding units or depth of the network.
        dropout (dict): Dictionary of layer arguments unpacked in `Dropout`.
        verbose (int): Level for print information.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    edge_pair_input = ks.layers.Input(**inputs[3])
    ed_pairs = edge_pair_input

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input

    # Make first edge hidden h0
    h_n0 = GatherNodesOutgoing()([n, edi])
    h0 = LazyConcatenate(axis=-1)([h_n0, ed])
    h0 = DenseEmbedding(**edge_initialize)(h0)

    # One Dense layer for all message steps
    edge_dense_all = DenseEmbedding(**edge_dense)  # Should be linear activation

    # Model Loop
    h = h0
    for i in range(depth):
        m_vw = DMPNNPPoolingEdgesDirected()([n, h, edi, ed_pairs])
        h = edge_dense_all(m_vw)
        h = LazyAdd()([h, h0])
        h = ActivationEmbedding(**edge_activation)(h)
        if dropout is not None:
            h = DropoutEmbedding(**dropout)(h)

    mv = PoolingLocalEdges(**pooling_args)([n, h, edi])
    mv = LazyConcatenate(axis=-1)([mv, n])
    hv = DenseEmbedding(**node_dense)(mv)

    # Output embedding choice
    n = hv
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_args)(n)
        # final dense layers
        main_output = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        main_output = GraphMLP(**output_mlp)(out)
        # no ragged for distribution supported atm
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(main_output)
    else:
        raise ValueError("Unsupported graph embedding for mode `DMPNN`")

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, edge_pair_input], outputs=main_output)
    return model
