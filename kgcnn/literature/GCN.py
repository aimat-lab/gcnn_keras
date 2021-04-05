import tensorflow.keras as ks

from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.layers.ragged.casting import ChangeIndexing
from kgcnn.layers.ragged.conv import GCN, DenseRagged
from kgcnn.layers.ragged.mlp import MLPRagged
from kgcnn.layers.ragged.pooling import PoolingNodes
from kgcnn.utils.models import generate_standard_graph_input, update_model_args


# 'Semi-Supervised Classification with Graph Convolutional Networks'
# by Thomas N. Kipf, Max Welling
# https://arxiv.org/abs/1609.02907
# https://github.com/tkipf/gcn


def make_gcn(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_embedd:dict = None,
        output_mlp:dict = None,
        # Model specific
        depth=3,
        gcn_args:dict = None,
        **kwargs):
    """
    Make GCN model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (dict):
        output_embedd (str): Graph or node embedding of the graph network. Default is 'graph'.

        output_mlp (dict): Use MLP for final output.
        
        depth (int, optional): Number of convolutions. Defaults to 3.
        gcn_args (dict):
        **kwargs

    Returns:
        model (tf.keras.models.Model): uncompiled model.

    """
    # Make default args
    input_embedd = update_model_args({"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
                                      "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
                                      "input_type": 'ragged'}, input_embedd)
    output_embedd = update_model_args({"output_mode": 'graph', "output_type": 'padded'},output_embedd)
    output_mlp = update_model_args({"use_bias" : [True, True, False], "units" : [25, 10, 1],
                                    "activation":['relu', 'relu', 'sigmoid']},output_mlp)
    gcn_args = update_model_args({"units" : 100, "use_bias" : True, "activation" : 'relu', "pooling_method":'segment_sum', "is_sorted":False,
                                "has_unconnected":"True"},gcn_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                                     input_edge_shape,
                                                                                                     None,
                                                                                                     **input_embedd)

    n = DenseRagged(gcn_args["units"], use_bias=True, activation='linear')(n) # To match units
    ed = ed
    edi = ChangeIndexing()([n, edge_index_input])

    # n-Layer Step
    for i in range(0, depth):
        n = GCN(node_indexing='batch',**gcn_args)([n, ed, edi])

    if output_embedd["output_mode"] == "graph":
        out = PoolingNodes()(n)  # will return tensor
        out = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = n
        out = MLPRagged(**output_mlp)(out)
        out = CastRaggedToDense()(out)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=out)

    return model
