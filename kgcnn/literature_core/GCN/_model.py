import keras_core as ks
from keras_core.layers import Dense, Activation
from kgcnn.layers_core.modules import Embedding
from kgcnn.layers_core.casting import CastBatchGraphListToPyGDisjoint
from kgcnn.layers_core.conv import GCN
# from kgcnn.layers_core.mlp import MLP
from kgcnn.layers_core.gather import GatherNodesOutgoing, GatherNodes
from kgcnn.layers_core.aggr import AggregateWeightedLocalEdges
from kgcnn.layers_core.pooling import PoolingNodes
from kgcnn.model.utils import update_model_kwargs

# Keep track of model version from commit date in literature.
__kgcnn_model_version__ = "2023.09.30"

# Supported backends
__kgcnn_model_backend_supported__ = ["tensorflow", "pytorch", "jax"]

# Implementation of GCN in `tf.keras` from paper:
# Semi-Supervised Classification with Graph Convolutional Networks
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn

model_default = {
    "name": "GCN",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32"},
               {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
               {"shape": (), "name": "node_num", "dtype": "int64"},
               {"shape": (), "name": "edge_num", "dtype": "int64"}],
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "input_edge_embedding": {"input_dim": 25, "output_dim": 1},
    "gcn_args": {"units": 100, "use_bias": True, "activation": "relu", "pooling_method": "sum"},
    "depth": 3,
    "verbose": 10,
    "output_embedding": "graph",
    "output_to_tensor": True,
    "output_mlp": {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ["relu", "relu", "sigmoid"]}
}


@update_model_kwargs(model_default, update_recursive=0)
def make_model(inputs: list = None,
               input_node_embedding: dict = None,
               input_edge_embedding: dict = None,
               depth: int = None,
               gcn_args: dict = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `GCN <https://arxiv.org/abs/1609.02907>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.GCN.model_default`.

    Inputs:
        list: `[node_attributes, edge_weights, edge_indices, ]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_weights (tf.RaggedTensor): Edge weights of shape `(batch, None, 1)`, that are entries of a scaled
              adjacency matrix.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_node_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        input_edge_embedding (dict): Dictionary of embedding arguments unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gcn_args (dict): Dictionary of layer arguments unpacked in :obj:`GCN` convolutional layer.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    if inputs[1]['shape'][-1] != 1:
        raise ValueError("No edge features available for GCN, only edge weights of pre-scaled adjacency matrix, \
                         must be shape (batch, None, 1), but got (without batch-dimension): %s." % inputs[1]['shape'])

    # Make input
    model_inputs = [ks.layers.Input(**x) for x in inputs]
    n, e, disjoint_indices, batch = CastBatchGraphListToPyGDisjoint()(model_inputs)

    # Embedding, if no feature dimension
    if len(inputs[0]['shape']) < 2:
        n = Embedding(**input_node_embedding)(n)
    if len(inputs[1]['shape']) < 2:
        e = Embedding(**input_edge_embedding)(e)

    # Model
    n = Dense(gcn_args["units"], use_bias=True, activation='linear')(n)  # Map to units

    for i in range(0, depth):
        # n = GCN(**gcn_args)([n, e, disjoint_indices])
        no = Dense(gcn_args["units"])(n)
        no = GatherNodesOutgoing()([no, disjoint_indices])
        nu = AggregateWeightedLocalEdges()([n, no, disjoint_indices, e])
        n = Activation(gcn_args["activation"])(nu)

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes()([n, batch])  # will return tensor
    elif output_embedding == "node":
        out = n
    else:
        raise ValueError("Unsupported output embedding for `GCN`")

    # out = MLP(**output_mlp)(n)
    out = Dense(1, activation="linear")(out)

    model = ks.models.Model(inputs=model_inputs, outputs=out)
    model.__kgcnn_model_version__ = __kgcnn_model_version__
    return model