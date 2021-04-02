import tensorflow.keras as ks

from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint, CastValuesToRagged
from kgcnn.layers.disjoint.conv import SchNetInteraction
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.disjoint.pooling import PoolingNodes
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.utils.activ import shifted_softplus
from kgcnn.utils.models import generate_standard_gaph_input


# Model Schnet as defined
# by Schuett et al. 2018
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566  
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


def getmodelSchnet(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = {},
        # Output
        output_embedd='graph',
        output_use_bias=[True, True, True],
        output_dim=[128, 64, 1],
        output_activation=['shifted_softplus', 'shifted_softplus', 'linear'],
        output_kernel_regularizer=[None, None, None],
        output_activity_regularizer=[None, None, None],
        output_bias_regularizer=[None, None, None],
        output_type='padded',
        # Model specific
        depth=4,
        node_dim=128,
        use_bias=True,
        activation='shifted_softplus',
        cfconv_pool="segment_sum",
        out_pooling_method="segment_sum",
        out_scale_pos=0,
        is_sorted=True,
        has_unconnected=False,
        **kwargs
):
    """
    Make uncompiled Schnet model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (list): Dictionary of input embedding info. See default values of kgcnn.utils.models.

        output_embedd (str): Graph or node embedding of the graph network. Default is 'graph'.
        output_use_bias (bool,list): Use bias for output. Optionally list for multiple layer.
                                     Default is [True,True,True].
        output_dim (list): Output dimension. Optionally list for multiple layer. Default is [32,16,1].
        output_activation (list): Activation function. Optionally list for multiple layer.
                                  Default is ['softplus2','softplus2','sigmoid'].
        output_kernel_regularizer (list): Kernel regularizer for output. Optionally list for multiple layer.
                                          Default is [None,None,None].
        output_activity_regularizer (list): Activity regularizer for output. Optionally list for multiple layer.
                                            Default is [None,None,None].
        output_bias_regularizer (list): Bias regularizer for output. Optionally list for multiple layer.
                                        Default is [None,None,None].
        output_type (str): Tensor output type. Default is 'padded'.


        depth (int, optional): Number of Interaction units. Defaults to 4.
        node_dim (int, optional): Hidden node dim. Defaults to 128.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str, optional): Activation function. Defaults to shifted_softplus.
        cfconv_pool (str, optional): Pooling method. Defaults to "segment_sum".
        out_pooling_method (str, optional): Node pooling method. Defaults to "segment_sum".
        out_scale_pos (int, optional): Scaling output, position of layer. Defaults to 0.
        is_sorted (bool, optional): Edge indices are sorted. Defaults to True.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to False.
        **kwargs

    Returns:
        model (tf.keras.models.Model): Schnet.

    """
    node_input, n, edge_input, ed, edge_index_input, _, _ = generate_standard_gaph_input(input_node_shape,
                                                                                         input_edge_shape, **input_embedd)

    n, node_len, ed, edge_len, edi = CastRaggedToDisjoint()([n, ed, edge_index_input])

    if isinstance(activation, str):
        if activation == 'shifted_softplus':
            activation = shifted_softplus

    if len(input_node_shape) > 1 and input_node_shape[-1] != node_dim:
        n = ks.layers.Dense(node_dim, activation='linear')(n)

    for i in range(0, depth):
        n = SchNetInteraction(node_dim, use_bias=use_bias, activation=activation, cfconv_pool=cfconv_pool,
                              is_sorted=is_sorted, has_unconnected=has_unconnected)(
            [n, node_len, ed, edge_len, edi])

    if len(output_dim) > 1:
        n = MLP(output_dim[:-1],
                mlp_use_bias=output_use_bias[:-1],
                mlp_activation=output_activation[:-1],
                mlp_activity_regularizer=output_kernel_regularizer[:-1],
                mlp_kernel_regularizer=output_kernel_regularizer[:-1],
                mlp_bias_regularizer=output_bias_regularizer[:-1])(n)

    mlp_last = ks.layers.Dense(output_dim[-1], activation=output_activation[-1], use_bias=output_use_bias[-1],
                               bias_regularizer=output_bias_regularizer[-1],
                               activity_regularizer=output_activity_regularizer[-1],
                               kernel_regularizer=output_kernel_regularizer[-1])

    if output_embedd == 'graph':
        if out_scale_pos == 0:
            n = mlp_last(n)
        out = PoolingNodes(pooling_method=out_pooling_method)([n, node_len])
        if out_scale_pos == 1:
            out = mlp_last(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = mlp_last(n)
        main_output = CastValuesToRagged()([out, node_len])
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model
