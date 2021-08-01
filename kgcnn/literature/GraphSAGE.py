import tensorflow as tf
import tensorflow.keras as ks
import pprint

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.keras import Concatenate, LayerNormalization
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pool.pooling import PoolingNodes, PoolingLocalMessages, PoolingLocalEdgesLSTM
from kgcnn.utils.models import generate_node_embedding, update_model_args, generate_edge_embedding


# 'Inductive Representation Learning on Large Graphs'
# William L. Hamilton and Rex Ying and Jure Leskovec
# http://arxiv.org/abs/1706.02216

def make_graph_sage(**kwargs):
    """Generate GraphSAGE network.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: GraphSAGE model.
    """
    model_args = kwargs
    model_default = {'name': "GraphSAGE",
                     'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                     'input_embedding': {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                         "edge_attributes": {"input_dim": 5, "output_dim": 64}},
                     'output_embedding': 'graph',
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'edge_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'pooling_args': {'pooling_method': "segment_mean"}, 'gather_args': {}, 'concat_args': {"axis": -1},
                     'use_edge_features': True, 'pooling_nodes_args':{'pooling_method': "mean"},
                     'depth': 3, 'verbose': 1
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO:kgcnn: Updated functional make model kwargs:")
        pprint.pprint(m)

    # Update default values
    inputs = m['inputs']
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    node_mlp_args = m['node_mlp_args']
    edge_mlp_args = m['edge_mlp_args']
    pooling_args = m['pooling_args']
    pooling_nodes_args = m['pooling_nodes_args']
    gather_args = m['gather_args']
    concat_args = m['concat_args']
    use_edge_features = m['use_edge_features']
    depth = m['depth']

    # Make input embedding, if no feature dimension
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])
    n = generate_node_embedding(node_input, inputs[0]['shape'], input_embedding[inputs[0]['name']])
    ed = generate_edge_embedding(edge_input, inputs[1]['shape'], input_embedding[inputs[1]['name']])
    edi = edge_index_input

    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu = GatherNodesOutgoing(**gather_args)([n, edi])
        if use_edge_features:
            eu = Concatenate(**concat_args)([eu, ed])

        eu = MLP(**edge_mlp_args)(eu)
        # Pool message
        if pooling_args['pooling_method'] in ["LSTM", "lstm"]:
            nu = PoolingLocalEdgesLSTM(**pooling_args)([n, eu, edi])
        else:
            nu = PoolingLocalMessages(**pooling_args)([n, eu, edi])  # Summing for each node connection

        nu = Concatenate(**concat_args)([n, nu])  # Concatenate node features with new edge updates

        n = MLP(**node_mlp_args)(nu)
        n = LayerNormalization(axis=-1)(n)  # Normalize

    # Regression layer on output
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_nodes_args)(n)
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be tensor
    elif output_embedding == 'node':
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `GraphSAGE`")

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model
