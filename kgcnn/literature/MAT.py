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
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "use_edge_embedding": False,
    "max_atoms": None,
    "distance_matrix_kwargs": {"trafo": "exp"},
    "attention_kwargs": {"units": 64, "lambda_a": 1.0, "lambda_g": 0.5, "lambda_d": 0.5},
    "feed_forward_kwargs": {"units": 64},
    "depth": 5,
    "heads": 8,
    "verbose": 10,
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                   "activation": ["kgcnn>softplus2", "kgcnn>softplus2", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               use_edge_embedding: bool = None,
               distance_matrix_kwargs: dict = None,
               attention_kwargs: dict = None,
               feed_forward_kwargs:dict = None,
               depth: int = None,
               heads: int = None,
               max_atoms: int = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `MAT <https://arxiv.org/pdf/2002.08264.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.MAT.model_default`.

    .. note::
        Please make sure to choose matching units for `attention_kwargs` and `feed_forward_kwargs`.

    Inputs:
        list: `[node_attributes, node_coordinates, edge_attributes, edge_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "MAT".
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level for print information.
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
    n = OptionalInputEmbedding(**input_embedding["node"], use_embedding=len(inputs[0]["shape"]) < 2)(node_input)
    ed = OptionalInputEmbedding(
        **input_embedding["edge"], use_embedding=len(inputs[1]["shape"]) < 2 and use_edge_embedding
    )(edge_input)
    edi = edge_index_input

    # Cast to dense Tensor with padding for MAT.
    # Nodes must have feature dimension.
    n, n_mask = ChangeTensorType(
        output_tensor_type="padded", shape=(None, max_atoms, None))(n)  # (batch, max_atoms, features)
    n_mask = MATReduceMask(axis=-1, keepdims=True)(n_mask)  # prefer broadcast mask (batch, max_atoms, 1)
    xyz, xyz_mask = ChangeTensorType(output_tensor_type="padded", shape=(None, max_atoms, 3))(xyz_input)
    dist, dist_mask = MATDistanceMatrix(**distance_matrix_kwargs)(
        xyz, mask=xyz_mask)  # Always be shape (batch, max_atoms, max_atoms, 1)
    adj, adj_mask = CastEdgeIndicesToDenseAdjacency(n_max=max_atoms)([ed, edi])  # (batch, max_atoms, max_atoms, (feat))

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
    h = ks.layers.Dense(attention_kwargs["units"], use_bias=False)(n)  # Assert correct feature dimension for skip.
    for _ in range(depth):
        # 1. Norm + Attention + Residual
        # TODO: Need to check padded Normalization.
        hn = ks.layers.LayerNormalization()(h)
        hn = ks.layers.Multiply()([hn, h_mask])

        hs = [
            MATAttentionHead(**attention_kwargs)(
                [hn, dist, adj],
                mask=[n_mask, dist_mask, dist_mask]
            )
            for _ in range(heads)
        ]  # Mask is applied in attention.
        hu = ks.layers.Add()(hs)  # Merge attention heads.
        h = ks.layers.Add()([h, hu])

        # 2. Norm + MLP + Residual
        # TODO: Need to check padded Normalization.
        hn = ks.layers.LayerNormalization()(h)
        hn = ks.layers.Multiply()([hn, h_mask])

        hu = MLP(**feed_forward_kwargs)(hn)
        hu = ks.layers.Multiply()([hu, h_mask])

        h = ks.layers.Add()([h, hu])

    # pooling output
    out = h
    out_mask = h_mask
    # TODO: Need to check padded Normalization.
    out = ks.layers.LayerNormalization()(out)
    out = ks.layers.Multiply()([out, out_mask])
    if output_embedding == 'graph':
        out = MATGlobalPool()(out, mask=out_mask)
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


# from kgcnn.data.datasets.ESOLDataset import ESOLDataset
# data = ESOLDataset()
# data.map_list(method= "normalize_edge_weights_sym")
# data.clean(model_default["inputs"])
# x_list = data.tensor(model_default["inputs"])
# model = make_model()
# out = model.predict(x_list)
