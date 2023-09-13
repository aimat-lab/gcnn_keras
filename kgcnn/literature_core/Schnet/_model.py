import keras_core as ks
from keras_core.layers import Dense
from kgcnn.layers_core.casting import (
    CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint, CastDisjointToGraph)
from kgcnn.layers_core.conv import SchNetInteraction
from kgcnn.layers_core.geom import NodeDistanceEuclidean, GaussBasisLayer, NodePosition, ShiftPeriodicLattice
from kgcnn.layers_core.modules import Embedding
from kgcnn.layers_core.mlp import MLP
from kgcnn.layers_core.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs
from keras_core.backend import backend as backend_to_use

# To be updated if model is changed in a significant way.
__model_version__ = "2023.09.07"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "torch", "jax"]
if backend_to_use() not in __kgcnn_model_backend_supported__:
    raise NotImplementedError("Backend '%s' for model 'GCN' is not supported." % backend_to_use())

# Implementation of Schnet in `keras` from paper:
# by Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela,
# Alexandre Tkatchenko, Klaus-Robert Müller (2018)
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


model_default = {
    "name": "Schnet",
    "inputs": [
        {"shape": (None,), "name": "node_attributes", "dtype": "float32"},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
        {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
        {"shape": (), "name": "total_nodes", "dtype": "int64"},
        {"shape": (), "name": "total_edges", "dtype": "int64"}
    ],
    "cast_disjoint_kwargs": {},
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "make_distance": True,
    "expand_distance": True,
    "interaction_args": {"units": 128, "use_bias": True,
                         "activation": "kgcnn>shifted_softplus", "cfconv_pool": "scatter_sum"},
    "node_pooling_args": {"pooling_method": "sum"},
    "depth": 4,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "verbose": 10,
    "last_mlp": {"use_bias": [True, True], "units": [128, 64],
                 "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": True,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["kgcnn>shifted_softplus", "linear"]}
}


@update_model_kwargs(model_default, update_recursive=0)
def make_model(inputs: list = None,
               cast_disjoint_kwargs: dict = None,
               input_node_embedding: dict = None,
               make_distance: bool = None,
               expand_distance: bool = None,
               gauss_args: dict = None,
               interaction_args: dict = None,
               node_pooling_args: dict = None,
               depth: int = None,
               name: str = None,
               verbose: int = None,
               last_mlp: dict = None,
               output_embedding: str = None,
               use_output_mlp: bool = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `SchNet <https://arxiv.org/abs/1706.08566>`__ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.Schnet.model_default` .

    Inputs:
        list: `[node_attributes, edge_distance, edge_indices, total_nodes, total_edges]`
        or `[node_attributes, node_coordinates, edge_indices, total_nodes, total_edges]`
        if :obj:`make_distance=True` and :obj:`expand_distance=True`
        to compute edge distances from node coordinates within the model.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, N, F)` or `(batch, N)`
              using an embedding layer.
            - edge_distance (tf.RaggedTensor): Edge distance of shape `(batch, M, D)` expanded
              in a basis of dimension `D` or `(batch, M, 1)` if using a :obj:`GaussBasisLayer` layer
              with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, M, 2)`.
            - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
            - total_nodes(Tensor, optional): Number of Nodes in graph if not same sized graphs of shape `(batch, )` .
            - total_edges(Tensor, optional): Number of Edges in graph if not same sized graphs of shape `(batch, )` .

    Outputs:
        Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        cast_indices_kwargs (dict): Dictionary of arguments for :obj:`CastBatchedIndicesToDisjoint` .
        input_node_embedding (dict): Dictionary of embedding arguments for nodes etc.
            unpacked in :obj:`Embedding` layers.
        make_distance (bool): Whether input is distance or coordinates at in place of edges.
        expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
            form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
        gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
        depth (int): Number of graph embedding units or depth of the network.
        interaction_args (dict): Dictionary of layer arguments unpacked in final :obj:`SchNetInteraction` layers.
        node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
        verbose (int): Level of verbosity.
        name (str): Name of the model.
        last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`ks.models.Model`
    """
    # Make input
    model_inputs = [ks.layers.Input(**x) for x in inputs]
    batched_nodes, batched_x, batched_indices, total_nodes, total_edges = model_inputs
    n, disjoint_indices, batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = CastBatchedIndicesToDisjoint(
        **cast_disjoint_kwargs)([batched_nodes, batched_indices, total_nodes, total_edges])

    # Optional Embedding.
    if len(inputs[0]['shape']) < 2:
        n = Embedding(**input_node_embedding)(n)

    if make_distance:
        x, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_x, total_nodes])
        pos1, pos2 = NodePosition()([x, disjoint_indices])
        ed = NodeDistanceEuclidean()([pos1, pos2])
    else:
        ed, _, _, _ = CastBatchedAttributesToDisjoint(**cast_disjoint_kwargs)([batched_x, total_edges])

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Model
    n = Dense(interaction_args["units"], activation='linear')(n)
    for i in range(0, depth):
        n = SchNetInteraction(**interaction_args)([n, ed, disjoint_indices])

    n = MLP(**last_mlp)(n)

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
        out = CastDisjointToGraph(**cast_disjoint_kwargs)(out)
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    else:
        raise ValueError("Unsupported output embedding for mode `SchNet` .")

    model = ks.models.Model(inputs=model_inputs, outputs=out)
    model.__kgcnn_model_version__ = __model_version__
    return model


# model_crystal_default = {
#     "name": "Schnet",
#     "inputs": [{"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
#                {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
#                {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
#                {"shape": (None, 3), "name": "edge_image", "dtype": "int64", "ragged": True},
#                {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False}],
#     "input_node_embedding": {"input_dim": 95, "output_dim": 64},
#     "make_distance": True, "expand_distance": True,
#     "interaction_args": {"units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"},
#     "node_pooling_args": {"pooling_method": "sum"},
#     "depth": 4,
#     "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
#     "verbose": 10,
#     "last_mlp": {"use_bias": [True, True], "units": [128, 64],
#                  "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},
#     "output_embedding": "graph", "output_to_tensor": True,
#     "use_output_mlp": True,
#     "output_mlp": {"use_bias": [True, True], "units": [64, 1],
#                    "activation": ["kgcnn>shifted_softplus", "linear"]}
# }
#
#
# @update_model_kwargs(model_default)
# def make_crystal_model(inputs: list = None,
#                        input_node_embedding: dict = None,
#                        make_distance: bool = None,
#                        expand_distance: bool = None,
#                        gauss_args: dict = None,
#                        interaction_args: dict = None,
#                        node_pooling_args: dict = None,
#                        depth: int = None,
#                        name: str = None,
#                        verbose: int = None,
#                        last_mlp: dict = None,
#                        output_embedding: str = None,
#                        use_output_mlp: bool = None,
#                        output_to_tensor: bool = None,
#                        output_mlp: dict = None
#                        ):
#     r"""Make `SchNet <https://arxiv.org/abs/1706.08566>`_ graph network via functional API.
#     Default parameters can be found in :obj:`kgcnn.literature.Schnet.model_crystal_default`.
#
#     Inputs:
#         list: `[node_attributes, edge_distance, edge_indices, edge_image, lattice]`
#         or `[node_attributes, node_coordinates, edge_indices, edge_image, lattice]` if :obj:`make_distance=True` and
#         :obj:`expand_distance=True` to compute edge distances from node coordinates within the model.
#
#             - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
#               using an embedding layer.
#             - edge_distance (tf.RaggedTensor): Edge distance of shape `(batch, None, D)` expanded
#               in a basis of dimension `D` or `(batch, None, 1)` if using a :obj:`GaussBasisLayer` layer
#               with model argument :obj:`expand_distance=True` and the numeric distance between nodes.
#             - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
#             - node_coordinates (tf.RaggedTensor): Node (atomic) coordinates of shape `(batch, None, 3)`.
#             - lattice (tf.Tensor): Lattice matrix of the periodic structure of shape `(batch, 3, 3)`.
#             - edge_image (tf.RaggedTensor): Indices of the periodic image the sending node is located. The indices
#                 of and edge are :math:`(i, j)` with :math:`j` being the sending node.
#
#     Outputs:
#         tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.
#
#     Args:
#         inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
#         input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
#         make_distance (bool): Whether input is distance or coordinates at in place of edges.
#         expand_distance (bool): If the edge input are actual edges or node coordinates instead that are expanded to
#             form edges with a gauss distance basis given edge indices. Expansion uses `gauss_args`.
#         gauss_args (dict): Dictionary of layer arguments unpacked in :obj:`GaussBasisLayer` layer.
#         depth (int): Number of graph embedding units or depth of the network.
#         interaction_args (dict): Dictionary of layer arguments unpacked in final :obj:`SchNetInteraction` layers.
#         node_pooling_args (dict): Dictionary of layer arguments unpacked in :obj:`PoolingNodes` layers.
#         verbose (int): Level of verbosity.
#         name (str): Name of the model.
#         last_mlp (dict): Dictionary of layer arguments unpacked in last :obj:`MLP` layer before output or pooling.
#         output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
#         use_output_mlp (bool): Whether to use the final output MLP. Possibility to skip final MLP.
#         output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
#         output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
#             Defines number of model outputs and activation.
#
#     Returns:
#         :obj:`tf.keras.models.Model`
#     """
#     # Make input
#     node_input = ks.layers.Input(**inputs[0])
#     xyz_input = ks.layers.Input(**inputs[1])
#     edge_index_input = ks.layers.Input(**inputs[2])
#     edge_image = ks.layers.Input(**inputs[3])
#     lattice = ks.layers.Input(**inputs[4])
#
#     # embedding, if no feature dimension
#     n = OptionalInputEmbedding(**input_embedding['node'],
#                                use_embedding=len(inputs[0]['shape']) < 2)(node_input)
#     edi = edge_index_input
#
#     if make_distance:
#         x = xyz_input
#         pos1, pos2 = NodePosition()([x, edi])
#         pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice])
#         ed = NodeDistanceEuclidean()([pos1, pos2])
#     else:
#         ed = xyz_input
#
#     if expand_distance:
#         ed = GaussBasisLayer(**gauss_args)(ed)
#
#     # Model
#     n = Dense(interaction_args["units"], activation='linear')(n)
#     for i in range(0, depth):
#         n = SchNetInteraction(**interaction_args)([n, ed, edi])
#
#     n = GraphMLP(**last_mlp)(n)
#
#     # Output embedding choice
#     if output_embedding == 'graph':
#         out = PoolingNodes(**node_pooling_args)(n)
#         if use_output_mlp:
#             out = MLP(**output_mlp)(out)
#     elif output_embedding == 'node':
#         out = n
#         if use_output_mlp:
#             out = GraphMLP(**output_mlp)(out)
#         if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
#             out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
#     else:
#         raise ValueError("Unsupported output embedding for mode `SchNet`")
#
#     model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input, edge_image, lattice], outputs=out)
#
#     model.__kgcnn_model_version__ = __model_version__
#     return model
