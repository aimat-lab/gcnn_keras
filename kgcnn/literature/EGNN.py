import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.geom import EuclideanNorm, NodePosition
from kgcnn.layers.modules import DenseEmbedding, OptionalInputEmbedding, LazyConcatenate, LazyMultiply, LazySubtract
from kgcnn.layers.gather import GatherEmbeddingSelection
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.utils.models import update_model_kwargs

ks = tf.keras

# Implementation of EGNN in `tf.keras` from paper:
# E(n) Equivariant Graph Neural Networks
# by Victor Garcia Satorras, Emiel Hoogeboom, Max Welling (2021)
# https://arxiv.org/abs/2102.09844


model_default = {
    "name": "Schnet",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "depth": 4,
    "verbose": 10,
    "node_pooling_kwargs": {"pooling_method": "sum"},
    "output_embedding": "graph", "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["kgcnn>shifted_softplus", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               use_edges: bool = None,
               edge_mlp_kwargs: dict = None,
               edge_attention_kwargs: dict = None,
               coord_mlp_kwargs: dict = None,
               name: str = None,
               verbose: int = None,
               node_pooling_kwargs: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `EGNN <https://arxiv.org/abs/2102.09844>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.EGNN.model_default`.

    Inputs:
        list: `[node_attributes, edge_distance, edge_indices]`
        or `[node_attributes, node_coordinates, edge_indices]` if :obj:`make_distance=True` and
        :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge distance of shape `(batch, None, D)` expanded
              in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
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
    edge_input = ks.layers.Input(**inputs[2])
    edge_index_input = ks.layers.Input(**inputs[3])

    # embedding, if no feature dimension
    h0 = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    edi = edge_index_input

    # Model
    h = h0
    x = node_input
    for i in range(0, depth):
        pos1, pos2 = NodePosition()([x, edi])
        rel_x = LazySubtract()([pos1, pos2])
        norm_x = EuclideanNorm()(rel_x)

        # Edge model
        h_i, h_j = GatherEmbeddingSelection()([h, edi])
        if use_edges:
            m_ij = LazyConcatenate()([h_i, h_j, norm_x])
        else:
            m_ij = LazyConcatenate()([h_i, h_j, rel_x, edge_input])
        if edge_mlp_kwargs:
            m_ij = GraphMLP(**edge_mlp_kwargs)(m_ij)
        if edge_attention_kwargs:
            m_att = GraphMLP(**edge_attention_kwargs)(m_ij)
            m_ij = LazyMultiply()([m_att, m_ij])

        # Coord model
        phi_m_ij = GraphMLP()(m_ij)
        x_trans = LazyMultiply()([])

    # Output embedding choice
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

    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input], outputs=out)
    return model
