import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType, ChangeIndexing
from kgcnn.layers.connect import AdjacencyPower
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.keras import Dense, Activation, Add
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingNodes, PoolingLocalEdges
from kgcnn.layers.topk import PoolingTopK, UnPoolingTopK
from kgcnn.ops.models import generate_standard_graph_input, update_model_args


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
                                      'input_tensor_type': 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'output_mlp': {"use_bias": [True, False], "units": [25, 1], "activation": ['relu', 'sigmoid']}
                     }

    # Update model args
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    pooling_args = {"pooling_method": 'segment_mean', "is_sorted": is_sorted, "has_unconnected": has_unconnected}

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, _, _ = generate_standard_graph_input(input_node_shape,
                                                                                          input_edge_shape, None,
                                                                                          **input_embedd)
    tens_type = "values_partition"
    node_indexing = "batch"
    n = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(n)
    ed = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(ed)
    edi = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(edge_index_input)
    edi = ChangeIndexing(input_tensor_type=tens_type, to_indexing=node_indexing)([n, edi])  # disjoint

    output_mlp.update({"input_tensor_type": tens_type})
    gather_args = {"input_tensor_type": tens_type, "node_indexing": node_indexing}
    pooling_args.update({"input_tensor_type": tens_type, "node_indexing": node_indexing})

    # Graph lists
    n = Dense(hidden_dim, use_bias=use_bias, activation='linear', input_tensor_type=tens_type)(n)
    in_graph = [n, ed, edi]
    graph_list = [in_graph]
    map_list = []

    # U Down
    i_graph = in_graph
    for i in range(0, depth):

        n, ed, edi = i_graph
        # GCN layer
        eu = GatherNodesOutgoing(**gather_args)([n, edi])
        eu = Dense(hidden_dim, use_bias=use_bias, activation='linear', input_tensor_type=tens_type)(eu)
        nu = PoolingLocalEdges(**pooling_args)([n, eu, edi])  # Summing for each node connection
        n = Activation(activation=activation, input_tensor_type=tens_type)(nu)

        if use_reconnect:
            ed, edi = AdjacencyPower(n=2, node_indexing=node_indexing, input_tensor_type=tens_type)([n, ed, edi])

        # Pooling
        i_graph, i_map = PoolingTopK(k=k, kernel_initializer=score_initializer,
                                     node_indexing=node_indexing, input_tensor_type=tens_type)([n, ed, edi])

        graph_list.append(i_graph)
        map_list.append(i_map)

    # U Up
    ui_graph = i_graph
    for i in range(depth, 0, -1):
        o_graph = graph_list[i - 1]
        i_map = map_list[i - 1]
        ui_graph = UnPoolingTopK(node_indexing=node_indexing, input_tensor_type=tens_type)(o_graph + i_map + ui_graph)

        n, ed, edi = ui_graph
        # skip connection
        n = Add(input_tensor_type=tens_type)([n, o_graph[0]])
        # GCN
        eu = GatherNodesOutgoing(**gather_args)([n, edi])
        eu = Dense(hidden_dim, use_bias=use_bias, activation='linear', input_tensor_type=tens_type)(eu)
        nu = PoolingLocalEdges(**pooling_args)([n, eu, edi])  # Summing for each node connection
        n = Activation(activation=activation, input_tensor_type=tens_type)(nu)

        ui_graph = [n, ed, edi]

    # Otuput
    n = ui_graph[0]
    if output_embedd["output_mode"] == 'graph':
        out = PoolingNodes(**pooling_args)(n)

        output_mlp.update({"input_tensor_type": "tensor"})
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type=tens_type, output_tensor_type="tensor")(out)

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model
