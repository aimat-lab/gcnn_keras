import tensorflow.keras as ks

from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint, CastValuesToRagged
from kgcnn.layers.disjoint.conv import SchNetInteraction
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.disjoint.pooling import PoolingNodes
from kgcnn.layers.ragged.casting import CastRaggedToDense
# from kgcnn.utils.activ import shifted_softplus
from kgcnn.utils.models import generate_standard_graph_input


# Model Schnet as defined
# by Schuett et al. 2018
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566  
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


def make_schnet(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_mlp: dict = None,
        output_dense: dict = None,
        output_embedd: dict = None,
        # Model specific
        depth=4,
        out_scale_pos=0,
        interaction_args: dict = None,
        node_pooling_args: dict = None,
        **kwargs
):
    """
    Make uncompiled SchNet model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (list): Dictionary of input embedding info. See default values of kgcnn.utils.models.

        output_mlp (dict, optional): Parameter for MLP output classification/ regression. Defaults to
                                    {"use_bias": [True, True], "units": [128, 64],
                                     "activation": ['shifted_softplus', 'shifted_softplus']}
        output_dense (dict): Parameter for Dense scaling layer. Defaults to {"units": 1, "activation": 'linear',
                             "use_bias": True}.
        output_embedd (str): Graph or node embedding of the graph network. Default is {"output_mode": 'graph'}.

        depth (int, optional): Number of Interaction units. Defaults to 4.
        interaction_args (dict): Interaction Layer arguments. Defaults include {"node_dim" : 128, "use_bias": True,
                                 "activation" : 'shifted_softplus', "cfconv_pool" : 'segment_sum',
                                 "is_sorted": False, "has_unconnected": True}
        node_pooling_args (dict, optional): Node pooling arguments. Defaults to {"pooling_method": "segment_sum"}.
        out_scale_pos (int, optional): Scaling output, position of layer. Defaults to 0.
        **kwargs

    Returns:
        model (tf.keras.models.Model): SchNet.

    """
    # Make default values if None
    input_embedd = {} if input_embedd is None else input_embedd
    interaction_args = {"node_dim": 128} if interaction_args is None else interaction_args
    output_mlp = {"mlp_use_bias": [True, True], "mlp_units": [128, 64],
                  "mlp_activation": ['shifted_softplus', 'shifted_softplus']} if output_mlp is None else output_mlp
    node_dim = interaction_args["node_dim"] if "node_dim" in interaction_args else 128  # Default
    output_dense = {"units": 1, "activation": 'linear', "use_bias": True} if output_dense is None else output_dense
    output_embedd = {"output_mode": 'graph', "output_type": 'padded'} if output_embedd is None else output_embedd
    node_pooling_args = {"pooling_method": "segment_sum"} if node_pooling_args is None else node_pooling_args

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, _, _ = generate_standard_graph_input(input_node_shape,
                                                                                          input_edge_shape, None,
                                                                                          **input_embedd)

    n, node_len, ed, edge_len, edi = CastRaggedToDisjoint()([n, ed, edge_index_input])

    if len(input_node_shape) > 1 and input_node_shape[-1] != node_dim:
        n = ks.layers.Dense(node_dim, activation='linear')(n)

    for i in range(0, depth):
        n = SchNetInteraction(**interaction_args)([n, node_len, ed, edge_len, edi])

    n = MLP(**output_mlp)(n)

    mlp_last = ks.layers.Dense(**output_dense)

    if output_embedd["output_mode"] == 'graph':
        if out_scale_pos == 0:
            n = mlp_last(n)
        out = PoolingNodes(**node_pooling_args)([n, node_len])
        if out_scale_pos == 1:
            out = mlp_last(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = mlp_last(n)
        main_output = CastValuesToRagged()([out, node_len])
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model
