import tensorflow as tf
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesSelection
from kgcnn.layers.modules import Dense, OptionalInputEmbedding, LazyMultiply, LazyAdd, Activation
from kgcnn.layers.pooling import PoolingLocalMessages
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.relational import RelationalDense
from kgcnn.model.utils import update_model_kwargs

ks = tf.keras

# Keep track of model version from commit date in literature.
# To be updated if model is changed in a significant way.
__model_version__ = "2022.11.25"

# Implementation of GCN in `tf.keras` from paper:
# GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation
# Marc Brockschmidt
# https://arxiv.org/abs/1906.12192


model_default = {
    "name": "GNNFilm",
    "inputs": [{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, ), "name": "edge_relations", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "dense_relation_kwargs": {"units": 64, "num_relations": 20},
    "dense_modulation_kwargs": {"units": 64, "num_relations": 20, "activation": "sigmoid"},
    "activation_kwargs": {"activation": "swish"},
    "depth": 3, "verbose": 10,
    "output_embedding": 'graph', "output_to_tensor": True,
    "output_mlp": {"use_bias": True, "units": 1,
                   "activation": "softmax"}
}


@update_model_kwargs(model_default)
def make_model(inputs: list = None,
               input_embedding: dict = None,
               depth: int = None,
               dense_relation_kwargs: dict = None,
               dense_modulation_kwargs: dict = None,
               activation_kwargs: dict = None,
               name: str = None,
               verbose: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `GNNFilm <https://arxiv.org/abs/1906.12192>`_ graph network via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.RGCN.model_default`.

    Inputs:
        list: `[node_attributes, edge_indices, edge_relations]`

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - edge_relations (tf.RaggedTensor): Edge relations of shape `(batch, None)` .

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        dense_relation_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalDense` layer.
        dense_modulation_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`RelationalDense` layer.
        activation_kwargs (dict):  Dictionary of layer arguments unpacked in :obj:`Activation` layer.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_index_input = ks.layers.Input(**inputs[1])
    edge_relations = ks.layers.Input(**inputs[2])

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    edi = edge_index_input

    # Model
    for i in range(0, depth):
        n_i, n_j = GatherNodesSelection(selection_index=[0, 1])([n, edi])
        # Note: This maybe could be done more efficiently.
        gamma = RelationalDense(**dense_modulation_kwargs)([n_i, edge_relations])
        beta = RelationalDense(**dense_modulation_kwargs)([n_i, edge_relations])
        h_j = RelationalDense(**dense_relation_kwargs)([n_j, edge_relations])
        m = LazyMultiply()([h_j, gamma])
        m = LazyAdd()([m, beta])
        h = PoolingLocalMessages(pooling_method="sum")([n, m, edi])
        n = Activation(**activation_kwargs)(h)

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes()(n)
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":  # Node labeling
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for mode `GNNFilm`")

    model = ks.models.Model(inputs=[node_input, edge_index_input, edge_relations], outputs=out, name=name)
    model.__kgcnn_model_version__ = __model_version__
    return model
