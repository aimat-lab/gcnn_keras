import tensorflow as tf
from math import inf
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.modules import LazyConcatenate, OptionalInputEmbedding, DenseEmbedding, ActivationEmbedding, ZerosLike
from kgcnn.layers.pooling import PoolingNodes, PoolingEmbeddingAttention
from kgcnn.layers.conv.hamnet_conv import HamNaiveDynMessage, HamNetFingerprintGenerator, HamNetGRUUnion, \
    HamNetNaiveUnion

# import tensorflow.keras as ks
# import tensorflow.python.keras as ks
ks = tf.keras

# Implementation of HamNet in `tf.keras` from paper:
# HamNet: Conformation-Guided Molecular Representation with Hamiltonian Neural Networks
# by Ziyao Li, Shuwen Yang, Guojie Song, Lingsheng Cai
# Link to paper: https://arxiv.org/abs/2105.03688
# Original implementation: https://github.com/PKUterran/HamNet
# Later implementation: https://github.com/PKUterran/MoleculeClub
# Note: the 2. implementation is cleaner than the original code and has been used as template.


hyper_model_default = {
    "name": "HamNet",
    "inputs": [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
               {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
               {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                        "edge": {"input_dim": 5, "output_dim": 64}},
    "message_kwargs": {"units": 128, "units_edge": 128},
    "fingerprint_kwargs": {"units": 128, "units_attend": 128, "depth": 2},
    "gru_kwargs": {"units": 128},
    "verbose": 10, "depth": 1,
    "union_type_node": "gru",
    "union_type_edge": "None",
    "given_coordinates": True,
    'output_embedding': 'graph', "output_to_tensor": True,
    'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                   "activation": ['relu', 'relu', 'linear']}
}


@update_model_kwargs(hyper_model_default, update_recursive=inf)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               verbose: int = None,
               message_kwargs: dict = None,
               gru_kwargs: dict = None,
               fingerprint_kwargs: dict = None,
               union_type_node: str = None,
               union_type_edge: str = None,
               given_coordinates: bool = None,
               depth: int = None,
               output_embedding: str = None,
               output_to_tensor: bool = None,
               output_mlp: dict = None
               ):
    r"""Make `HamNet <https://arxiv.org/abs/2105.03688>`_ graph model via functional API.
    Default parameters can be found in :obj:`kgcnn.literature.HamNet.hyper_model_default`.

    .. note::
        At the moment only the Fingerprint Generator for graph embeddings is implemented and coordinates must
        be provided as model input.

    Inputs:
        list: `[node_attributes, edge_attributes, edge_indices]`,
        or `[node_attributes, edge_attributes, edge_indices, node_coordinates]` if :obj:`given_coordinates=True`.

            - node_attributes (tf.RaggedTensor): Node attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_attributes (tf.RaggedTensor): Edge attributes of shape `(batch, None, F)` or `(batch, None)`
              using an embedding layer.
            - edge_indices (tf.RaggedTensor): Index list for edges of shape `(batch, None, 2)`.
            - node_coordinates (tf.RaggedTensor): Euclidean coordinates of nodes of shape `(batch, None, 3)`.

    Outputs:
        tf.Tensor: Graph embeddings of shape `(batch, L)` if :obj:`output_embedding="graph"`.

    Args:
        name (str): Name of the model.
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict):  Dictionary of embedding arguments for nodes etc. unpacked in :obj:`Embedding` layers.
        verbose (int): Level of verbosity. For logging and printing.
        message_kwargs (dict): Dictionary of layer arguments unpacked in message passing layer for node updates.
        gru_kwargs (dict): Dictionary of layer arguments unpacked in gated recurrent unit update layer.
        fingerprint_kwargs (dict): Dictionary of layer arguments unpacked in :obj:`HamNetFingerprintGenerator` layer.
        given_coordinates (bool): Whether coordinates are provided as model input, or are computed by the Model.
        union_type_edge (str): Union type of edge updates. Choose "gru", "naive" or "None".
        union_type_node (str): Union type of node updates. Choose "gru", "naive" or "None".
        depth (int): Depth or number of (message passing) layers of the model.
        output_embedding (str): Main embedding task for graph network. Either "node", "edge" or "graph".
        output_to_tensor (bool): Whether to cast model output to :obj:`tf.Tensor`.
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification :obj:`MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # Make input embedding if no feature dimension. (batch, None) -> (batch, None, F)
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input

    # Generate coordinates.
    if given_coordinates:
        # Case for given coordinates.
        q_ftr = ks.layers.Input(**inputs[3])
        p_ftr = ZerosLike()(q_ftr)
    else:
        # Use Hamiltonian engine to get p, q coordinates.
        raise NotImplementedError("Hamiltonian engine not yet implemented")

    # Initialization
    n = DenseEmbedding(units=gru_kwargs["units"], activation="tanh")(n)
    ed = DenseEmbedding(units=gru_kwargs["units"], activation="tanh")(ed)
    p = p_ftr
    q = q_ftr

    # Message passing.
    for i in range(depth):
        # Message step
        nu, eu = HamNaiveDynMessage(**message_kwargs)([n, ed, p, q, edi])

        # Node updates
        if union_type_node == "gru":
            n = HamNetGRUUnion(**gru_kwargs)([n, nu])
        elif union_type_node == "naive":
            n = HamNetNaiveUnion(units=gru_kwargs["units"])([n, nu])
        else:
            n = nu

        # Edge updates
        if union_type_edge == "gru":
            ed = HamNetGRUUnion(**gru_kwargs)([ed, eu])
        elif union_type_edge == "naive":
            ed = HamNetNaiveUnion(units=gru_kwargs["units"])([ed, eu])
        else:
            ed = eu

    # Fingerprint generator for graph embedding.
    if output_embedding == 'graph':
        out = HamNetFingerprintGenerator(**fingerprint_kwargs)(n)
        out = ks.layers.Flatten()(out)  # will be tensor.
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        if output_to_tensor:  # For tf version < 2.8 cast to tensor below.
            out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported output embedding for `HamNet`")

    # Make Model instance.
    if given_coordinates:
        model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, q_ftr], outputs=out)
    else:
        model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)
    return model
