import tensorflow.keras as ks
import pprint
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.schnet_conv import SchNetInteraction
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.geom import NodeDistance, GaussBasisLayer
from kgcnn.layers.pool.pooling import PoolingNodes
from kgcnn.utils.models import update_model_args, generate_node_embedding


# Model Schnet as defined
# by Schuett et al. 2018
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566  
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


def make_model(**kwargs):
    """Make un-compiled SchNet model.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: SchNet.
    """
    model_args = kwargs
    model_default = {'name': "SchNet",
                     'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                     'input_embedding': {"node_attributes": {"input_dim": 95, "output_dim": 64}},
                     'output_embedding': 'graph',
                     'interaction_args': {"units": 128, "use_bias": True,
                                          "activation": 'kgcnn>shifted_softplus', "cfconv_pool": 'sum',
                                          "is_sorted": False, "has_unconnected": True},
                     'output_mlp': {"use_bias": [True, True], "units": [128, 64],
                                    "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus']},
                     'output_dense': {"units": 1, "activation": 'linear', "use_bias": True},
                     'node_pooling_args': {"pooling_method": "sum"},
                     'depth': 4, 'out_scale_pos': 0,
                     'gauss_ags': {"bins": 20, "range": 4, "offset": 0.0, "sigma": 0.4},
                     'verbose': 1
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO:kgcnn: Updated functional make model kwargs:")
        pprint.pprint(m)

    # Update args
    inputs = m['inputs']
    input_embedding = m['input_embedding']
    interaction_args = m['interaction_args']
    output_mlp = m['output_mlp']
    output_dense = m['output_dense']
    output_embedding = m['output_embedding']
    node_pooling_args = m['node_pooling_args']
    depth = m['depth']
    out_scale_pos = m['out_scale_pos']
    gauss_ags = m['gauss_ags']

    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # embedding, if no feature dimension
    n = generate_node_embedding(node_input, inputs[0]['shape'], input_embedding[inputs[0]['name']])
    edi = edge_index_input
    x = xyz_input
    ed = NodeDistance()([x, edi])
    ed = GaussBasisLayer(**gauss_ags)(ed)

    # Model
    n = Dense(interaction_args["units"], activation='linear')(n)
    for i in range(0, depth):
        n = SchNetInteraction(**interaction_args)([n, ed, edi])

    n = MLP(**output_mlp)(n)
    mlp_last = Dense(**output_dense)

    # Output embedding choice
    if output_embedding["output_mode"] == 'graph':
        if out_scale_pos == 0:
            n = mlp_last(n)
        out = PoolingNodes(**node_pooling_args)(n)
        if out_scale_pos == 1:
            out = mlp_last(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = mlp_last(n)
        main_output = ChangeTensorType(input_tensor_type="values_partition", output_tensor_type="tensor")(out)  # no ragged for distribution atm

    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input], outputs=main_output)
    return model
