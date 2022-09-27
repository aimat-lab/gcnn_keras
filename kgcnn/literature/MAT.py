import tensorflow as tf
from typing import Union
from kgcnn.layers.modules import OptionalInputEmbedding
from kgcnn.layers.casting import ChangeTensorType, CastEdgeIndicesToDenseAdjacency
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.conv.mat_conv import MATAttentionHead, MATDistanceMatrix, MATReduceMask, MATGlobalPool, MATExpandMask

ks = tf.keras

# Implementation of MAT in `tf.keras` from paper:
# Molecule Attention Transformer
# Łukasz Maziarka, Tomasz Danel, Sławomir Mucha, Krzysztof Rataj, Jacek Tabor, Stanisław Jastrzębski
# https://arxiv.org/abs/2002.08264
# https://github.com/ardigen/MAT
# https://github.com/lucidrains/molecule-attention-transformer


model_default = {
    "name": "MAT",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 1), "name": "edge_weights", "dtype": "float32", "ragged": True},  # or edge_weights
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 95, "output_dim": 64}},
    "use_edge_embedding": False,
    "max_atoms": None,
    "distance_matrix_kwargs": {"trafo": "exp"},
    "attention_kwargs": {"units": 8, "lambda_attention": 0.3, "lambda_distance": 0.3, "lambda_adjacency": None,
                         "dropout": 0.1},
    "feed_forward_kwargs": {"units": [32, 32, 32], "activation": ["relu", "relu", "linear"]},
    "embedding_units": 32,
    "depth": 5,
    "heads": 8,
    "merge_heads": "concat",
    "verbose": 10,
    "pooling_kwargs": {"pooling_method": "sum"},
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                   "activation": ["relu", "relu", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               use_edge_embedding: bool = None,
               distance_matrix_kwargs: dict = None,
               attention_kwargs: dict = None,
               feed_forward_kwargs:dict = None,
               embedding_units: int = None,
               depth: int = None,
               heads: int = None,
               merge_heads: str = None,
               max_atoms: int = None,
               verbose: int = None,
               pooling_kwargs: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `MAT <https://arxiv.org/pdf/2002.08264.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.MAT.model_default`.

    .. note::
        We added a linear layer to keep correct node embedding dimension.

    Inputs:
        list: `[node_attributes, node_coordinates, edge_attributes, edge_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "MAT".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level for print information.
        use_edge_embedding (bool): Whether to use edge input embedding regardless of edge input shape.
        distance_matrix_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`MATDistanceMatrix`.
        attention_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`MATDistanceMatrix`.
        feed_forward_kwargs (dict): Dictionary of layer arguments unpacked in feed forward :obj:`MLP`.
        embedding_units (int): Units for node embedding.
        heads (int): Number of attention heads
        merge_heads (str): How to merge head, using either 'sum' or 'concat'.
        max_atoms (int): Fixed (maximum) number of atoms for padding. Can be `None`.
        pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`MATGlobalPool`.
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

    # Embedding, if no feature dimension
    nd = OptionalInputEmbedding(**input_embedding["node"], use_embedding=len(inputs[0]["shape"]) < 2)(node_input)
    ed = OptionalInputEmbedding(
        **input_embedding["edge"], use_embedding=len(inputs[1]["shape"]) < 2 and use_edge_embedding
    )(edge_input)
    edi = edge_index_input

    # Cast to dense Tensor with padding for MAT.
    # Nodes must have feature dimension.
    n, n_mask_f = ChangeTensorType(
        output_tensor_type="padded", shape=(None, max_atoms, None))(nd)  # (batch, max_atoms, features)
    n_mask = MATReduceMask(axis=-1, keepdims=True)(n_mask_f)  # prefer broadcast mask (batch, max_atoms, 1)
    xyz, xyz_mask = ChangeTensorType(output_tensor_type="padded", shape=(None, max_atoms, 3))(xyz_input)
    # Always has shape (batch, max_atoms, max_atoms, 1)
    dist, dist_mask = MATDistanceMatrix(**distance_matrix_kwargs)(
        xyz, mask=xyz_mask)
    # Adjacency matrix padded (batch, max_atoms, max_atoms, (features))
    adj, adj_mask = CastEdgeIndicesToDenseAdjacency(n_max=max_atoms)([nd, ed, edi])

    # Check shapes
    # print(n.shape, dist.shape, adj.shape)
    # print(n_mask.shape, dist_mask.shape, adj_mask.shape)

    # Adjacency is derived from edge input. If edge input has no last dimension and no embedding is used, then adjacency
    # matrix will have shape (batch, max_atoms, max_atoms) and edge input should be ones or weights or bond degree.
    # Otherwise, adjacency bears feature expanded from edge attributes of shape (batch, max_atoms, max_atoms, features).
    has_edge_dim = len(inputs[1]["shape"]) >= 2 or len(inputs[1]["shape"]) < 2 and use_edge_embedding

    if has_edge_dim:
        # Assume that feature-wise attention is not desired for adjacency, reduce to single value.
        adj = ks.layers.Dense(1, use_bias=False)(adj)
        adj_mask = MATReduceMask(axis=-1, keepdims=True)(adj_mask)
    else:
        # Make sure that shape is (batch, max_atoms, max_atoms, 1).
        adj = MATExpandMask(axis=-1)(adj)
        adj_mask = MATExpandMask(axis=-1)(adj_mask)

    # Repeat for depth.
    h_mask = n_mask
    h = ks.layers.Dense(units=embedding_units, use_bias=False)(n)  # Assert correct feature dimension for skip.
    for _ in range(depth):
        # 1. Norm + Attention + Residual
        hn = ks.layers.LayerNormalization()(h)
        hs = [
            MATAttentionHead(**attention_kwargs)(
                [hn, dist, adj],
                mask=[n_mask, dist_mask, adj_mask]
            )
            for _ in range(heads)
        ]
        if merge_heads in ["add", "sum", "reduce_sum"]:
            hu = ks.layers.Add()(hs)
            hu = ks.layers.Dense(units=embedding_units, use_bias=False)(hu)
        else:
            hu = ks.layers.Concatenate(axis=-1)(hs)
            hu = ks.layers.Dense(units=embedding_units, use_bias=False)(hu)
        h = ks.layers.Add()([h, hu])

        # 2. Norm + MLP + Residual
        hn = ks.layers.LayerNormalization()(h)
        hu = MLP(**feed_forward_kwargs)(hn)
        hu = ks.layers.Dense(units=embedding_units, use_bias=False)(hu)
        hu = ks.layers.Multiply()([hu, h_mask])
        h = ks.layers.Add()([h, hu])

    # pooling output
    out = h
    out_mask = h_mask
    out = ks.layers.LayerNormalization()(out)
    if output_embedding == 'graph':
        out = ks.layers.Multiply()([out, out_mask])
        out = MATGlobalPool(**pooling_kwargs)(out, mask=out_mask)
        # final prediction MLP for the output!
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(out)
        out = ks.layers.Multiply()([out, out_mask])
    else:
        raise ValueError("Unsupported graph embedding for mode `MAT`")

    model = ks.models.Model(
        inputs=[node_input, xyz_input, edge_input, edge_index_input],
        outputs=out,
        name=name
    )
    return model
