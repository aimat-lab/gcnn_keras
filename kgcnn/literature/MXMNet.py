import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.geom import NodeDistanceEuclidean, EdgeAngle, BesselBasisLayer, NodePosition, ShiftPeriodicLattice
from kgcnn.layers.modules import DenseEmbedding, OptionalInputEmbedding, LazyConcatenate, LazySubtract, LazyAdd
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.conv.dimenet_conv import SphericalBasisLayer

ks = tf.keras

# Implementation of MXMNet in `tf.keras` from paper:
# Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures
# by Shuo Zhang, Yang Liu, Lei Xie (2020)
# https://arxiv.org/abs/2011.07457
# https://github.com/zetayue/MXMNet


model_default = {
    "name": "MXMNet",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, ), "name": "edge_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "bessel_basis_local": {"num_radial": 16, "cutoff": 3.0, "envelope_exponent": 5},
    "bessel_basis_global": {"num_radial": 16, "cutoff": 6.0, "envelope_exponent": 5},
    "spherical_basis_local": {"num_spherical": 7, "num_radial": 6, "cutoff": 5.0, "envelope_exponent": 5},
    "use_edge_attributes": False,
    "depth": 4,
    "verbose": 10,
    "node_pooling_args": {"pooling_method": "sum"},
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               node_pooling_args: dict = None,
               depth: int = None,
               name: str = None,
               bessel_basis_local: dict = None,
               bessel_basis_global: dict = None,
               spherical_basis_local: dict = None,
               use_edge_attributes: bool = None,
               verbose: int = None,
               output_embedding: str = None,
               use_output_mlp: bool = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `MXMNet <https://github.com/zetayue/MXMNet>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.MXMNet.model_default`.

    Inputs:
        list: `[node_attributes, node_coordinates, edge_attributes, edge_indices, angle_indices, range_indices]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - range_indices (tf.RaggedTensor): Index list for range-like edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - angle_indices (tf.RaggedTensor): Index list of angles referring to range connections of
              shape `(batch, None, 2)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
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
    angle_index_input = ks.layers.Input(**inputs[4])
    range_index_input = ks.layers.Input(**inputs[5])

    # Rename to short names and make embedding, if no feature dimension.
    x = xyz_input
    n = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'], use_embedding=len(inputs[2]['shape']) < 2)(edge_input)
    ei_l = edge_index_input
    ri_g = range_index_input
    ai_1 = angle_index_input

    # Calculate distances and spherical and bessel basis for local edges including angles.
    # For the first version, we restrict ourselves to 2-hop angles.
    pos1_l, pos2_l = NodePosition()([x, ei_l])
    d_l = NodeDistanceEuclidean()([pos1_l, pos2_l])
    rbf_l = BesselBasisLayer(**bessel_basis_local)(d_l)
    v12_l = LazySubtract()([pos1_l, pos2_l])
    a_l = EdgeAngle()([v12_l, ai_1])
    sbf_l = SphericalBasisLayer(**spherical_basis_local)([d_l, a_l, ai_1])

    # Calculate distance and bessel basis for global (range) edges.
    pos1_g, pos2_g = NodePosition()([x, ri_g])
    d_g = NodeDistanceEuclidean()([pos1_g, pos2_g])
    rbf_g = BesselBasisLayer(**bessel_basis_global)(d_g)

    if use_edge_attributes:
        rbf_l = LazyConcatenate()([rbf_l, ed])

    rbf_l = GraphMLP()(rbf_l)
    sbf_l = GraphMLP()(sbf_l)
    rbf_g = GraphMLP()(rbf_g)

    # Model
    h = n
    nodes_list = [n]
    for i in range(0, depth):

        nodes_list.append(h)

    # Output embedding choice
    out = LazyAdd()(nodes_list)
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)(out)
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)(out)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `MXMNet`")

    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input], outputs=out, name=name)
    return model
