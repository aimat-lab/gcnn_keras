import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.schnet_conv import SchNetInteraction
from kgcnn.layers.geom import NodeDistance, GaussBasisLayer
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pool.pooling import PoolingNodes
from kgcnn.utils.models import update_model_kwargs, generate_embedding

# Model Schnet as defined
# by Schuett et al. 2018
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566  
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


model_default = {'name': "Schnet",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64}},
                 'output_embedding': 'graph',
                 'interaction_args': {"units": 128, "use_bias": True,
                                      "activation": 'kgcnn>shifted_softplus', "cfconv_pool": 'sum'},
                 'output_mlp': {"use_bias": [True, True], "units": [128, 64],
                                "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus']},
                 'output_dense': {"units": 1, "activation": 'linear', "use_bias": True},
                 'node_pooling_args': {"pooling_method": "sum"},
                 'depth': 4, 'out_scale_pos': 0,
                 'gauss_ags': {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
                 'verbose': 1
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               interaction_args=None,
               output_mlp=None,
               output_dense=None,
               output_embedding=None,
               node_pooling_args=None,
               depth=None,
               out_scale_pos=None,
               gauss_ags=None,
               **kwargs):
    """Make SchNet graph model."""
    # Make input
    node_input = ks.layers.Input(**inputs[0])
    xyz_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # embedding, if no feature dimension
    n = generate_embedding(node_input, inputs[0]['shape'], input_embedding['node'])
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
    if output_embedding == 'graph':
        if out_scale_pos == 0:
            n = mlp_last(n)
        out = PoolingNodes(**node_pooling_args)(n)
        if out_scale_pos == 1:
            out = mlp_last(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    elif output_embedding == 'node':
        out = mlp_last(n)
        main_output = ChangeTensorType(input_tensor_type="values_partition", output_tensor_type="tensor")(
            out)  # no ragged for distribution atm
    else:
        raise ValueError("Unsupported graph embedding for mode `SchNet`")

    model = ks.models.Model(inputs=[node_input, xyz_input, edge_index_input], outputs=main_output)
    return model
