import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.modules import LazyConcatenate, OptionalInputEmbedding, DenseEmbedding, ActivationEmbedding, ZerosLike
from kgcnn.layers.pooling import PoolingNodes, PoolingEmbeddingAttention
from kgcnn.layers.conv.mpnn_conv import GRUUpdate
from kgcnn.layers.conv.hamnet_conv import HamNaiveDynMessage, HamNetFingerprintGenerator

# Model by
# HamNet: Conformation-Guided Molecular Representation with Hamiltonian Neural Networks
# Ziyao Li, Shuwen Yang, Guojie Song, Lingsheng Cai
# https://arxiv.org/abs/2105.03688
# Original implementation: https://github.com/PKUterran/HamNet


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
                       "verbose": 10, "depth": 1, "use_coordinates": True,
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
               use_coordinates: bool = True,
               depth: int = None,
               output_embedding: str = None,
               output_mlp: dict = None
               ):
    """Under Construction!!!!"""
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # Make input embedding if no feature dimension. (batch, None) -> (batch, None, F)
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    ed = OptionalInputEmbedding(**input_embedding['edge'],
                                use_embedding=len(inputs[1]['shape']) < 2)(edge_input)
    edi = edge_index_input

    if use_coordinates:
        q_ftr = ks.layers.Input(**inputs[3])
        p_ftr = ZerosLike()(q_ftr)
    else:
        raise NotImplementedError("Hamiltonian engine not yet implemented")

    # Second part of HamNet. Attentive Message passing. Very Similar to Attentive FP.
    n = DenseEmbedding(units=gru_kwargs["units"], activation="tanh")(n)
    ed = DenseEmbedding(units=gru_kwargs["units"], activation="tanh")(ed)
    p = p_ftr
    q = q_ftr
    for i in range(depth):
        nu, eu = HamNaiveDynMessage(**message_kwargs)([n, ed, p, q, edi])
        n = GRUUpdate(**gru_kwargs)([n, nu])
        ed = GRUUpdate(**gru_kwargs)([ed, eu])
        # New edge, in original version MLP, later also GRU.

    # Fingerprint generator for graph embedding.
    if output_embedding == 'graph':
        out = HamNetFingerprintGenerator(**fingerprint_kwargs)(n)
        out = ks.layers.Flatten()(out)  # will be tensor
        main_output = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type='ragged',
                                       output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for `HamNet`")

    if use_coordinates:
        model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input, q_ftr],
                                      outputs=main_output)
    else:
        model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input],
                                      outputs=main_output)
    return model
