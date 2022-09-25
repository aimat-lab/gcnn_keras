import tensorflow as tf
from typing import Union
from kgcnn.layers.modules import OptionalInputEmbedding
from kgcnn.layers.casting import ChangeTensorType, CastEdgeIndicesToDenseAdjacency
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.conv.mat_conv import MATAttentionHead, MATDistanceMatrix

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
        {"shape": (None,), "name": "edge_number", "dtype": "float32", "ragged": True},  # or edge_weights
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "use_explicit_edge_embedding": False,
    "max_atoms": None,
    "verbose": 10,
    "depth": 5,
    "units": 64,
    "heads": 8,
    "output_embedding": "graph",
    "output_to_tensor": True,
}


@update_model_kwargs(model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               use_explicit_edge_embedding: bool = None,
               depth: int = None,
               units: int = None,
               heads: int = None,
               max_atoms: int = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               ):
    r"""Make `MAT <https://arxiv.org/pdf/2002.08264.pdf>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.MAT.model_default`.

    .. note::
        Please make sure to choose matching dimensions for .

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
        **input_embedding["edge"], use_embedding=len(inputs[1]["shape"]) < 2 and use_explicit_edge_embedding
    )(node_input)
    edi = edge_index_input

    # Cast to dense Tensor with padding for MAT.
    # Nodes must have feature dimension.
    n, n_mask = ChangeTensorType(output_tensor_type="padded", shape=(None, max_atoms, None))(n)
    xyz, xyz_mask = ChangeTensorType(output_tensor_type="padded", shape=(None, max_atoms, 3))(xyz_input)
    dist, dist_mask = MATDistanceMatrix()(xyz, mask=xyz_mask)  # Always be shape (batch, max_atoms, max_atoms)
    adj, adj_mask = CastEdgeIndicesToDenseAdjacency(n_max=max_atoms)([ed, edi])  # (batch, max_atoms, max_atoms, feat)

    # Adjacency is derived from edge input. If edge input has no last dimension and no embedding is used, then adjacency
    # matrix will have shape (batch, max_atoms, max_atoms) and edge input should be ones or weights or bond degree.
    # Otherwise, adjacency bears feature expanded from edge attributes of shape (batch, max_atoms, max_atoms, features).

    # depth loop
    h = n
    for _ in range(depth):
        # part one Norm + Attention + Residual
        hn = ks.layers.LayerNormalization()(h)
        hs = [
            MATAttentionHead(units=units)(
                [hn, dist, adj],
                mask=[n_mask, dist_mask, xyz_mask]
            )
            for _ in range(heads)
        ]
        hn = ks.layers.Add()(hs)
        # part two Norm + MLP + Residual
        hn = ks.layers.LayerNormalization()(hn)
        h += MLP(units=units)(hn)

    # pooling output
    out = h
    if output_embedding == 'graph':
        out = ks.layers.LayerNormalization()(out)
        # mean pooling can be a parameter
        out = tf.math.reduce_mean(out, axis=-2)
        # final prediction MLP for the output!
        out = MLP(units=units)(out)
    elif output_embedding == 'node':
        pass
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
# data.clean(model_default["inputs"])
# x_list = data.tensor(model_default["inputs"])
# model = make_model()
# out = model.predict(x_list)
