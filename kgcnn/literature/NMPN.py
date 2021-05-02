import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType, ChangeIndexing
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.set2set import Set2Set
from kgcnn.layers.update import GRUupdate, TrafoMatMulMessages
from kgcnn.ops.models import generate_standard_graph_input, update_model_args


# Neural Message Passing for Quantum Chemistry
# by Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
# http://arxiv.org/abs/1704.01212    


def make_nmpn(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific
        depth=3,
        node_dim=128,
        edge_dense: dict = None,
        use_set2set=True,
        set2set_args: dict = None,
        pooling_args: dict = None
):
    """
    Get Message passing model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
            'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
            'input_type': 'ragged'}
        output_embedd (str): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_type": 'padded'}
        output_mlp (dict): Dictionary of MLP arguments for output regression or classifcation. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "output_activation": ['selu', 'selu', 'sigmoid']}
        depth (int, optional): Depth. Defaults to 3.
        node_dim (int, optional): Dimension for hidden node representation. Defaults to 128.
        edge_dense (dict): Dictionary of arguments for NN to make edge matrix. Default is
            {'use_bias' : True, 'activation' : 'selu'}
        use_set2set (bool, optional): Use set2set layer. Defaults to True.
        set2set_args (dict): Dictionary of Set2Set Layer Arguments. Default is
            {'channels': 32, 'T': 3, "pooling_method": "sum", "init_qstar": "0"}
        pooling_args (dict): Dictionary for message pooling arguments. Default is
            {'is_sorted': False, 'has_unconnected': True, 'pooling_method': "segment_mean"}

    Returns:
        model (ks.models.Model): Message Passing model.
    """
    # Make default parameter
    model_default = {'input_embedd': {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
                                      'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
                                      'input_tensor_type': 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['selu', 'selu', 'sigmoid']},
                     'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "sum",
                                      "init_qstar": "0"},
                     'pooling_args': {'is_sorted': False, 'has_unconnected': True, 'pooling_method': "segment_mean"},
                     'edge_dense': {'use_bias': True, 'activation': 'selu'}
                     }

    # Update model args
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    set2set_args = update_model_args(model_default['set2set_args'], set2set_args)
    pooling_args = update_model_args(model_default['pooling_args'], pooling_args)
    edge_dense = update_model_args(model_default['edge_dense'], edge_dense)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, _, _ = generate_standard_graph_input(input_node_shape,
                                                                                          input_edge_shape,
                                                                                          None,
                                                                                          **input_embedd)

    tens_type = "values_partition"
    node_indexing = "batch"
    n = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(n)
    ed = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(ed)
    edi = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(edge_index_input)
    edi = ChangeIndexing(input_tensor_type=tens_type, to_indexing=node_indexing)([n, edi])
    set2set_args.update({"input_tensor_type": tens_type})
    output_mlp.update({"input_tensor_type": tens_type})
    edge_dense.update({"input_tensor_type": tens_type})
    pooling_args.update({"input_tensor_type": tens_type, "node_indexing": node_indexing})

    n = Dense(node_dim, activation="linear", input_tensor_type=tens_type)(n)
    edge_net = Dense(node_dim * node_dim, **edge_dense)(ed)
    gru = GRUupdate(node_dim, input_tensor_type=tens_type, node_indexing=node_indexing)

    for i in range(0, depth):
        eu = GatherNodesOutgoing(input_tensor_type=tens_type, node_indexing=node_indexing)([n, edi])
        eu = TrafoMatMulMessages(node_dim, input_tensor_type=tens_type, node_indexing=node_indexing)([edge_net, eu])
        eu = PoolingLocalEdges(**pooling_args)(
            [n, eu, edi])  # Summing for each node connections
        n = gru([n, eu])

    if output_embedd["output_mode"] == 'graph':
        if use_set2set:
            # output
            outss = Dense(set2set_args['channels'], activation="linear", input_tensor_type=tens_type)(n)
            out = Set2Set(**set2set_args)(outss)
        else:
            out = PoolingNodes(**pooling_args)(n)

        # final dense layers
        output_mlp.update({"input_tensor_type": "tensor"})
        main_output = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = n
        main_output = MLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type=tens_type, output_tensor_type="tensor")(main_output)
        # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model
