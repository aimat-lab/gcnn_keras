import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate, ActivationEmbedding, LazyAdd, DropoutEmbedding, \
    OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.conv.dmpnn_conv import DMPNNPPoolingEdgesDirected
from kgcnn.utils.models import update_model_kwargs

ks = tf.keras

# Implementation of DMPNN in `tf.keras` from paper:
# Analyzing Learned Molecular Representations for Property Prediction
# by Kevin Yang, Kyle Swanson, Wengong Jin, Connor Coley, Philipp Eiden, Hua Gao,
# Angel Guzman-Perez, Timothy Hopper, Brian Kelley, Miriam Mathea, Andrew Palmer,
# Volker Settels, Tommi Jaakkola, Klavs Jensen, and Regina Barzilay
# https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237

model_default = {
    "name": "DMPNN",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None,), "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64},
                        "graph": {"input_dim": 100, "output_dim": 64}},
    "pooling_args": {"pooling_method": "sum"},
    "use_graph_state": False,
    "edge_initialize": {"units": 128, "use_bias": True, "activation": "relu"},
    "edge_dense": {"units": 128, "use_bias": True, "activation": "linear"},
    "edge_activation": {"activation": "relu"},
    "node_dense": {"units": 128, "use_bias": True, "activation": "relu"},
    "verbose": 10, "depth": 5, "dropout": {"rate": 0.1},
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               pooling_args: dict = None,
               edge_initialize: dict = None,
               edge_dense: dict = None,
               edge_activation: dict = None,
               node_dense: dict = None,
               dropout: dict = None,
               depth: int = None,
               verbose: int = None,
               use_graph_state: bool = False,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.DMPNN.model_default`.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices, edge_pairs]` or
        `[node_attributes, edge_attributes, edge_indices, edge_pairs, state_attributes]` if `use_graph_state=True`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_pairs (tf.RaggedTensor): Pair mappings for reverse edge for each edge `(batch, None, 1)`.
            - state_attributes (tf.Tensor): Environment or graph state attributes of shape `(batch, F)` or `(batch,)`
              using an embedding layer.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "DMPNN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes`,
            :obj:`PoolingLocalEdges` layers.
        edge_initialize (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for first edge embedding.
        edge_dense (dict): Dictionary of layer arguments unpacked in :obj:`Dense` layer for edge embedding.
        edge_activation (dict): Edge Activation after skip connection.
        node_dense (dict): Dense kwargs for node embedding layer.
        depth (int): Number of graph embedding units or depth of the network.
        dropout (dict): Dictionary of layer arguments unpacked in :obj:`Dropout`.
        verbose (int): Level for print information.
        use_graph_state (bool): Whether to use graph state information. Default is False.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    edge_pair_input = ks.layers.Input(**inputs[3])
    graph_state_input = ks.layers.Input(**inputs[4]) if use_graph_state else None

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(
        **input_embedding["node"],
        use_embedding=len(inputs[0]["shape"]) < 2)(node_input)
    ed = OptionalInputEmbedding(
        **input_embedding["edge"],
        use_embedding=len(inputs[1]["shape"]) < 2)(edge_input)
    graph_state = OptionalInputEmbedding(
        **input_embedding["graph"],
        use_embedding=len(inputs[4]["shape"]) < 1)(graph_state_input) if use_graph_state else None
    edi = edge_index_input
    ed_pairs = edge_pair_input

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
        if use_graph_state:
            out = ks.layers.Concatenate()([graph_state, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if use_graph_state:
            graph_state_node = GatherState()([graph_state, n])
            n = LazyConcatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `DMPNN`.")

    if use_graph_state:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, edge_pair_input, graph_state_input],
            outputs=out, name=name)
    else:
        model = ks.models.Model(
            inputs=[node_input, edge_input, edge_index_input, edge_pair_input],
            outputs=out, name=name)
    return model
