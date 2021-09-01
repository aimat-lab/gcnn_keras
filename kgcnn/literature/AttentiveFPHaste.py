import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.attention import AttentiveHeadFP
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.utils.models import generate_embedding, update_model_kwargs
from kgcnn.layers.conv.haste import HasteLayerNormGRUUpdate, HastePoolingNodesAttentiveLayerNorm

# Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism
# Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li,
# Xiaomin Luo, Kaixian Chen, Hualiang Jiang*, and Mingyue Zheng*
# Cite this: J. Med. Chem. 2020, 63, 16, 8749–8760
# Publication Date:August 13, 2019
# https://doi.org/10.1021/acs.jmedchem.9b00959

model_default = {'name': "AttentiveFP",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64},
                                     "edge": {"input_dim": 5, "output_dim": 64}},
                 'output_embedding': 'graph',
                 'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                "activation": ['relu', 'relu', 'sigmoid']},
                 'attention_args': {"units": 32},
                 'depth': 3,
                 'dropout': 0.1,
                 'verbose': 1
                 }


@update_model_kwargs(model_default)
def make_model_haste(inputs=None,
                     depth=None,
                     dropout=None,
                     input_embedding=None,
                     output_embedding=None,
                     output_mlp=None,
                     attention_args=None,
                     **kwargs):
    """Make AttentiveFP graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Note: This implementation uses GRU-cells from haste.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        depth (int): Number of graph embedding units or depth of the network.
        dropout (float): Dropout to use.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.
        attention_args (dict): Dictionary of layer arguments unpacked in `AttentiveHeadFP` layer. Units parameter
            is also used in GRU-update and `PoolingNodesAttentive`.

    Returns:
        tf.keras.models.Model
    """

    #  Make input
    node_input = ks.layers.Input(**inputs[0])
    edge_input = ks.layers.Input(**inputs[1])
    edge_index_input = ks.layers.Input(**inputs[2])

    # Embedding, if no feature dimension
    n = generate_embedding(node_input, inputs[0]['shape'], input_embedding['node'])
    ed = generate_embedding(edge_input, inputs[1]['shape'], input_embedding['edge'])
    edi = edge_index_input

    # Model
    nk = Dense(units=attention_args['units'])(n)
    ck = AttentiveHeadFP(use_edge_features=True, **attention_args)([nk, ed, edi])
    nk = HasteLayerNormGRUUpdate(units=attention_args['units'], dropout=dropout)([nk, ck])

    for i in range(1, depth):
        ck = AttentiveHeadFP(**attention_args)([nk, ed, edi])
        nk = HasteLayerNormGRUUpdate(units=attention_args['units'], dropout=dropout)([nk, ck])
    n = nk

    # Output embedding choice
    if output_embedding == 'graph':
        out = HastePoolingNodesAttentiveLayerNorm(units=attention_args['units'], dropout=dropout)(n)
        output_mlp.update({"input_tensor_type": "tensor"})
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    elif output_embedding == 'node':  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `AttentiveFP`")

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model
