import tensorflow as tf
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.modules import LazyConcatenate, OptionalInputEmbedding, DenseEmbedding, ActivationEmbedding, ZerosLike
from kgcnn.layers.pooling import PoolingNodes, PoolingEmbeddingAttention
from kgcnn.layers.conv.hamnet_conv import HamNaiveDynMessage, HamNetFingerprintGenerator, HamNetGRUUnion, HamNetNaiveUnion
# import tensorflow.keras as ks
# import tensorflow.python.keras as ks
ks = tf.keras


# Implementation of HamNet in `tf.keras` from paper:
# HamNet: Conformation-Guided Molecular Representation with Hamiltonian Neural Networks
# by Ziyao Li, Shuwen Yang, Guojie Song, Lingsheng Cai
# Link to paper: https://arxiv.org/abs/2105.03688
# Original implementation: https://github.com/PKUterran/HamNet
# Later implementation: https://github.com/PKUterran/MoleculeClub
# Note: the 2. implementation is cleaner than the original code.


hyper_model_default = {"name": "HamNet",
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
                       'output_embedding': 'graph',
                       'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                      "activation": ['relu', 'relu', 'linear']}
                       }


@update_model_kwargs(hyper_model_default)
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
               output_mlp: dict = None
               ):
    """Make HamNet graph model via functional API. Default parameters can be found in :obj:`hyper_model_default`.
    At the moment only the Fingerprint Generator for graph embeddings is implemented and coordinates must be provided
    as model input.

    Args:
        name (str):
        inputs (list):
        input_embedding (dict):
        verbose (int):
        message_kwargs (dict):
        gru_kwargs (dict):
        fingerprint_kwargs (dict):
        given_coordinates (bool):
        union_type_edge (str):
        union_type_node (str):
        depth (int):
        output_embedding (str):
        output_mlp (dict):

    Returns:
        tf.keras.models.Model
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
        main_output = MLP(**output_mlp)(out)

    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type='ragged',
                                       output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for `HamNet`")

    # Make Model instance.
    if given_coordinates:
        model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input, q_ftr],
                                      outputs=main_output)
    else:
        model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input],
                                      outputs=main_output)
    return model
