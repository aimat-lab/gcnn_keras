import tensorflow.keras as ks

from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint, CastValuesToRagged
from kgcnn.layers.disjoint.connect import AdjacencyPower
from kgcnn.layers.disjoint.gather import GatherNodesOutgoing
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.disjoint.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.disjoint.topk import PoolingTopK, UnPoolingTopK
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.layers.ragged.conv import DenseRagged
from kgcnn.utils.models import generate_standard_graph_input, update_model_args


# Graph U-Nets
# by Hongyang Gao, Shuiwang Ji
# https://arxiv.org/pdf/1905.05178.pdf


def make_unet(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific
        hidden_dim=32,
        depth=4,
        k=0.3,
        score_initializer='ones',
        use_bias=True,
        activation='relu',
        is_sorted=False,
        has_unconnected=True,
        use_reconnect=True
):
    """
    Make Graph U Net.
    
    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (list): Dictionary of embedding parameters used if input shape is None. Default is
            {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
            'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
            'input_type': 'ragged'}
        output_mlp (dict, optional): Parameter for MLP output classification/ regression. Defaults to
            {"use_bias": [True, False], "output_dim": [25, 1],
            "activation": ['relu', 'sigmoid']}
        output_embedd (str): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_type": 'padded'}
        hidden_dim (int): Hidden node feature dimension 32,
        depth (int): Depth of pooling steps. Default is 4.
        k (float): Pooling ratio. Default is 0.3.
        score_initializer (str): How to initialize score kernel. Default is 'ones'.
        use_bias (bool): Use bias. Default is True.
        activation (str): Activation function used. Default is 'relu'.
        is_sorted (bool, optional): Edge edge_indices are sorted. Defaults to False.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to True.
        use_reconnect (bool): Reconnect nodes after pooling. I.e. adj_matrix=adj_matrix^2. Default is True.
    
    Returns:
        model (ks.models.Model): Unet model.
    """
    # Default values update
    model_default = {'input_embedd': {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
                                      'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
                                      'input_type': 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'output_mlp': {"use_bias": [True, False], "units": [25, 1], "activation": ['relu', 'sigmoid']}
                     }

    # Update model args
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, _, _ = generate_standard_graph_input(input_node_shape,
                                                                                          input_edge_shape, None,
                                                                                          **input_embedd)

    n = DenseRagged(hidden_dim, use_bias=use_bias, activation='linear')(n)

    n, node_len, ed, edge_len, edi = CastRaggedToDisjoint()([n, ed, edge_index_input])

    in_graph = [n, node_len, ed, edge_len, edi]

    graph_list = [in_graph]
    map_list = []

    # U Down
    i_graph = in_graph
    for i in range(0, depth):

        n, node_len, ed, edge_len, edi = i_graph
        # GCN layer
        eu = GatherNodesOutgoing()([n, node_len, edi, edge_len])
        eu = ks.layers.Dense(hidden_dim, use_bias=use_bias, activation='linear')(eu)
        nu = PoolingLocalEdges(pooling_method='segment_mean', is_sorted=is_sorted, has_unconnected=has_unconnected)(
            [n, node_len, eu, edge_len, edi])  # Summing for each node connection
        n = ks.layers.Activation(activation=activation)(nu)

        if use_reconnect:
            edi, ed, edge_len = AdjacencyPower(n=2)([edi, ed, edge_len, node_len])

        # Pooling
        i_graph, i_map = PoolingTopK(k=k, kernel_initializer=score_initializer)([n, node_len, ed, edge_len, edi])

        graph_list.append(i_graph)
        map_list.append(i_map)

    # U Up
    ui_graph = i_graph
    for i in range(depth, 0, -1):
        o_graph = graph_list[i - 1]
        i_map = map_list[i - 1]
        ui_graph = UnPoolingTopK()(o_graph + i_map + ui_graph)

        n, node_len, ed, edge_len, edi = ui_graph
        # skip connection
        n = ks.layers.Add()([n, o_graph[0]])
        # GCN
        eu = GatherNodesOutgoing()([n, node_len, edi, edge_len])
        eu = ks.layers.Dense(hidden_dim, use_bias=use_bias, activation='linear')(eu)
        nu = PoolingLocalEdges(pooling_method='segment_mean', is_sorted=is_sorted, has_unconnected=has_unconnected)(
            [n, node_len, eu, edge_len, edi])  # Summing for each node connection
        n = ks.layers.Activation(activation=activation)(nu)

        ui_graph = [n, node_len, ed, edge_len, edi]

    # Otuput
    n = ui_graph[0]
    node_len = ui_graph[1]
    if output_embedd["output_mode"] == 'graph':
        out = PoolingNodes(pooling_method='segment_mean')([n, node_len])

        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = CastValuesToRagged()([out, node_len])
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model
