import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.geom import EuclideanNorm, NodePosition, EdgeDirectionNormalized, PositionEncodingBasisLayer
from kgcnn.layers.modules import LazyAdd, OptionalInputEmbedding, LazyConcatenate, LazyMultiply, LazySubtract
from kgcnn.layers.gather import GatherEmbeddingSelection
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes, PoolingLocalEdges
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.norm import GraphLayerNormalization

ks = tf.keras

# Implementation of EGNN in `tf.keras` from paper:
# E(n) Equivariant Graph Neural Networks
# by Victor Garcia Satorras, Emiel Hoogeboom, Max Welling (2021)
# https://arxiv.org/abs/2102.09844


model_default = {
    "name": "EGNN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 10), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "depth": 4,
    "node_mlp_initialize": None,
    "use_edge_attributes": True,
    "edge_mlp_kwargs": {"units": [64, 64], "activation": ["swish", "linear"]},
    "edge_attention_kwargs": None,  # {"units: 1", "activation": "sigmoid"}
    "use_normalized_difference": False,
    "expand_distance_kwargs": None,
    "coord_mlp_kwargs":  {"units": [64, 1], "activation": ["swish", "linear"]},  # option: "tanh" at the end.
    "pooling_coord_kwargs": {"pooling_method": "mean"},
    "pooling_edge_kwargs": {"pooling_method": "sum"},
    "node_normalize_kwargs": None,
    "node_mlp_kwargs": {"units": [64, 64], "activation": ["swish", "linear"]},
    "use_skip": True,
    "verbose": 10,
    "node_pooling_kwargs": {"pooling_method": "sum"},
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               node_mlp_initialize: dict = None,
               use_edge_attributes: bool = None,
               edge_mlp_kwargs: dict = None,
               edge_attention_kwargs: dict = None,
               use_normalized_difference: bool = None,
               expand_distance_kwargs: dict = None,
               coord_mlp_kwargs: dict = None,
               pooling_coord_kwargs: dict = None,
               pooling_edge_kwargs: dict = None,
               node_normalize_kwargs: dict = None,
               node_mlp_kwargs: dict = None,
               use_skip: bool = None,
               verbose: int = None,
               node_pooling_kwargs: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `EGNN <https://arxiv.org/abs/2102.09844>`_ graph network via functional API.

    Default parameters can be found in :obj:`kgcnn.literature.EGNN.model_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, edge_attributes, edge_indices]`
        or `[node_attributes, node_coordinates, edge_indices]` if :obj:`use_edge_attributes=False`.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, D)`.
                Can also be ignored if not needed.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Default is "EGNN".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        node_mlp_initialize (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer for start embedding.
        use_edge_attributes (bool): Whether to use edge attributes including for example further edge information.
        edge_mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        edge_attention_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        use_normalized_difference (bool): Whether to use a normalized difference vector for nodes.
        expand_distance_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PositionEncodingBasisLayer`.
        coord_mlp_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphMLP` layer.
        pooling_coord_kwargs (dict):
        pooling_edge_kwargs (dict):
        node_normalize_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`GraphLayerNormalization` layer.
        node_mlp_kwargs (dict):
        use_skip (bool):
        verbose (int): Level of verbosity.
        node_pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    if use_edge_attributes:
        edge_input = ks.layers.Input(**inputs[2])
        edge_index_input = ks.layers.Input(**inputs[3])
    else:
        edge_input = None
        edge_index_input = ks.layers.Input(**inputs[2])

    # embedding, if no feature dimension
    h0 = OptionalInputEmbedding(
        **input_embedding['node'],
        use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    edi = edge_index_input

    # Model
    h = GraphMLP(**node_mlp_initialize)(h0) if node_mlp_initialize else h0
    x = node_input
    for i in range(0, depth):
        pos1, pos2 = NodePosition()([x, edi])
        diff_x = LazySubtract()([pos1, pos2])
        norm_x = EuclideanNorm()(diff_x)
        # Original code as normalize option for coord-differences
        if use_normalized_difference:
            diff_x = EdgeDirectionNormalized()([pos1, pos2])
        if expand_distance_kwargs:
            norm_x = PositionEncodingBasisLayer()(norm_x)

        # Edge model
        h_i, h_j = GatherEmbeddingSelection()([h, edi])
        if use_edge_attributes:
            m_ij = LazyConcatenate()([h_i, h_j, norm_x, edge_input])
        else:
            m_ij = LazyConcatenate()([h_i, h_j, norm_x])
        if edge_mlp_kwargs:
            m_ij = GraphMLP(**edge_mlp_kwargs)(m_ij)
        if edge_attention_kwargs:
            m_att = GraphMLP(**edge_attention_kwargs)(m_ij)
            m_ij = LazyMultiply()([m_att, m_ij])

        # Coord model
        if coord_mlp_kwargs:
            m_ij_weights = GraphMLP(**coord_mlp_kwargs)(m_ij)
            x_trans = LazyMultiply()([m_ij_weights, diff_x])
            agg = PoolingLocalEdges(**pooling_coord_kwargs)([h, x_trans, edi])
            x = LazyAdd()([x, agg])

        # Node model
        m_i = PoolingLocalEdges(**pooling_edge_kwargs)([h, m_ij, edi])
        if node_mlp_kwargs:
            m_i = LazyConcatenate()([h, m_i])
            m_i = GraphMLP(**node_mlp_kwargs)(m_i)
        if node_normalize_kwargs:
            h = GraphLayerNormalization(**node_normalize_kwargs)(h)
        if use_skip:
            h = LazyAdd()([h, m_i])
        else:
            h = m_i

    # Output embedding choice
    n = h
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_kwargs)(n)
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `SchNet`")

    if use_edge_attributes:
        model = ks.models.Model(inputs=[node_input, xyz_input, edge_input, edge_index_input], outputs=out, name=name)
    else:
        model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input], outputs=out, name=name)
    return model
