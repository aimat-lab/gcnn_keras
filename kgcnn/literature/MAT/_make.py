import keras as ks
from keras.backend import backend as backend_to_use
from kgcnn.layers.modules import Embedding
from kgcnn.layers.mlp import MLP
from kgcnn.models.utils import update_model_kwargs
from ._layers import MATAttentionHead, MATDistanceMatrix, MATReduceMask, MATGlobalPool, MATExpandMask

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2023-12-08"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'MAT' is not supported." % backend_to_use())

# Implementation of MAT in `tf.keras` from paper:
# Molecule Attention Transformer
# Łukasz Maziarka, Tomasz Danel, Sławomir Mucha, Krzysztof Rataj, Jacek Tabor, Stanisław Jastrzębski
# https://arxiv.org/abs/2002.08264
# https://github.com/ardigen/MAT
# https://github.com/lucidrains/molecule-attention-transformer


model_default = {
    "name": "MAT",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, None, 1), "name": "adjacency_matrix", "dtype": "float32"},
        {"shape": (None,), "name": "node_mask", "dtype": "bool"},
        {"shape": (None, None), "name": "adjacency_mask", "dtype": "bool"},
    ],
    "input_tensor_type": "padded",
    "input_embedding": None,
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 95, "output_dim": 64},
    "max_atoms": None,
    "distance_matrix_kwargs": {"trafo": "exp"},
    "attention_kwargs": {"units": 8, "lambda_attention": 0.3, "lambda_distance": 0.3, "lambda_adjacency": None,
                         "dropout": 0.1, "add_identity": False},
    "feed_forward_kwargs": {"units": [32, 32, 32], "activation": ["relu", "relu", "linear"]},
    "embedding_units": 32,
    "depth": 5,
    "heads": 8,
    "merge_heads": "concat",
    "verbose": 10,
    "pooling_kwargs": {"pooling_method": "sum"},
    "output_embedding": "graph",
    "output_to_tensor": None,
    "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                   "activation": ["relu", "relu", "linear"]},
    "output_tensor_type": "padded"
}


@update_model_kwargs(model_default, update_recursive=0, deprecated=["input_embedding", "output_to_tensor"])
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               input_node_embedding: dict = None,
               input_tensor_type: str = None,
               input_edge_embedding: dict = None,
               distance_matrix_kwargs: dict = None,
               attention_kwargs: dict = None,
               max_atoms: int = None,
               feed_forward_kwargs:dict = None,
               embedding_units: int = None,
               depth: int = None,
               heads: int = None,
               merge_heads: str = None,
               verbose: int = None,
               pooling_kwargs: dict = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None,
               output_tensor_type: str = None
               ):
    r"""Make `MAT <https://arxiv.org/pdf/2002.08264.pdf>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.MAT.model_default` .

    .. note::

        We added a linear layer to keep correct node embedding dimension.

    Inputs:
        list: `[node_attributes, node_coordinates, adjacency_matrix, node_mask, adjacency_mask]`

            - node_attributes (Tensor): Node attributes of shape `(batch, N, F)` or `(batch, N)`
              using an embedding layer.
            - node_coordinates (Tensor): Node (atomic) coordinates of shape `(batch, N, 3)`.
            - adjacency_matrix (Tensor): Edge attributes of shape `(batch, N, N, F)` or `(batch, N, N)`
              using an embedding layer.
            - node_mask (Tensor): Node mask of shape `(batch, N)`
            - adjacency_mask (Tensor): Adjacency mask of shape `(batch, N, N)`

    Outputs:
        Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model. Should be "MAT".
        inputs (list): List of dictionaries unpacked in :obj:`keras.layers.Input`. Order must match model definition.
        input_tensor_type (str): Input tensor type. Only "padded" is valid for this implementation.
        input_node_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level for print information.
        distance_matrix_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`MATDistanceMatrix`.
        attention_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`MATDistanceMatrix`.
        feed_forward_kwargs (dict): Dictionary of layer arguments unpacked in feed forward :obj:`MLP`.
        embedding_units (int): Units for node embedding.
        heads (int): Number of attention heads
        merge_heads (str): How to merge head, using either 'sum' or 'concat'.
        pooling_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`MATGlobalPool`.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.
        output_tensor_type (str): Output tensor type. Only "padded" is valid for this implementation.

    Returns:
        :obj:`keras.models.Model`
    """
    assert input_tensor_type in ["padded", "mask", "masked"], "Only padded tensors are valid for this implementation."
    assert output_tensor_type in ["padded", "mask", "masked"], "Only padded tensors are valid for this implementation."
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    adjacency_matrix = ks.layers.Input(**inputs[2])
    node_mask = ks.layers.Input(**inputs[3])
    adjacency_mask = ks.layers.Input(**inputs[4])

    use_edge_embedding = input_edge_embedding is not None
    use_node_embedding = input_node_embedding is not None
    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(node_input)
    else:
        n = node_input
    if use_edge_embedding:
        adj = Embedding(**input_edge_embedding)(adjacency_matrix)
    else:
        adj = adjacency_matrix
    n_mask = node_mask
    adj_mask = adjacency_mask
    xyz = xyz_input
    n_mask = MATExpandMask(axis=-1)(n_mask)
    adj_mask = MATExpandMask(axis=-1)(adj_mask)

    # Cast to dense Tensor with padding for MAT.
    # Nodes must have feature dimension.
    dist, dist_mask = MATDistanceMatrix(**distance_matrix_kwargs)(xyz, mask=n_mask)

    # Check shapes
    # print(n.shape, dist.shape, adj.shape)
    # print(n_mask.shape, dist_mask.shape, adj_mask.shape)

    # Adjacency is derived from edge input. If edge input has no last dimension and no embedding is used, then adjacency
    # matrix will have shape (batch, max_atoms, max_atoms) and edge input should be ones or weights or bond degree.
    # Otherwise, adjacency bears feature expanded from edge attributes of shape (batch, max_atoms, max_atoms, features).
    has_edge_dim = len(inputs[2]["shape"]) >= 3 or len(inputs[2]["shape"]) < 3 and use_edge_embedding

    if has_edge_dim:
        # Assume that feature-wise attention is not desired for adjacency, reduce to single value.
        adj = ks.layers.Dense(1, use_bias=False)(adj)
    else:
        # Make sure that shape is (batch, max_atoms, max_atoms, 1).
        adj = MATExpandMask(axis=-1)(adj)

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
        out = MLP(**output_mlp)(out)
        out = ks.layers.Multiply()([out, out_mask])
    else:
        raise ValueError("Unsupported graph embedding for mode `MAT` .")

    model = ks.models.Model(
        inputs=[node_input, xyz_input, adjacency_matrix, node_mask, adjacency_mask],
        outputs=out,
        name=name
    )
    model.__kgcnn_model_version__ = __model_version__
    return model
