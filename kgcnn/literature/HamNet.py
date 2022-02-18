import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.utils.models import update_model_kwargs
from kgcnn.layers.modules import LazyConcatenate, OptionalInputEmbedding, DenseEmbedding, ActivationEmbedding
from kgcnn.layers.pooling import PoolingNodes, PoolingEmbeddingAttention
from kgcnn.layers.conv.mpnn_conv import GRUUpdate
from kgcnn.layers.conv.hamnet_conv import HamNaiveDynMessage, HamNetFingerprintGenerator

# Model by
# HamNet: Conformation-Guided Molecular Representation with Hamiltonian Neural Networks
# Ziyao Li, Shuwen Yang, Guojie Song, Lingsheng Cai
# https://arxiv.org/abs/2105.03688
# Original implementation: https://github.com/PKUterran/HamNet


hyper_model_default = {'name': "HamNet",
                       'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                  {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                  {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                       'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                           "edge": {"input_dim": 5, "output_dim": 64}},
                       "verbose": 10, "depth": 3,
                       'output_embedding': 'graph',
                       'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                      "activation": ['relu', 'relu', 'sigmoid']}
                       }


@update_model_kwargs(hyper_model_default)
def make_model(name: str = None,
               inputs: list = None,
               input_embedding: dict = None,
               verbose: int = None,
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

    # Second part of HamNet. Fingerprint generator. Very Similar to AttentiveFP.
    n, ed, p, q
    for i in range(depth):
        nu, eu = HamNaiveDynMessage()([n, ed, p, q, edi])
        n = GRUUpdate()([n, nu])
        ed = GRUUpdate()([ed, eu])
        # New edge, in original version MLP, later also GRU.

    if output_embedding == 'graph':
        out = HamNetFingerprintGenerator()(n)
        out = ks.layers.Flatten()(out)  # will be tensor
        main_output = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type='ragged',
                                       output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for `HamNet`")

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input],
                                  outputs=main_output)
    return model
