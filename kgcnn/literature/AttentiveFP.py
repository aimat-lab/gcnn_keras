import tensorflow as tf
import tensorflow.keras as ks
import pprint

from kgcnn.layers.attention import AttentiveHeadFP, PoolingNodesAttentive
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.keras import Dense, Dropout
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.mlp import MLP
from kgcnn.utils.models import generate_node_embedding, update_model_args, generate_edge_embedding


# Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism
# Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li, Xiaomin Luo, Kaixian Chen, Hualiang Jiang*, and Mingyue Zheng*
# Cite this: J. Med. Chem. 2020, 63, 16, 8749–8760
# Publication Date:August 13, 2019
# https://doi.org/10.1021/acs.jmedchem.9b00959


def make_attentiveFP(**kwargs):
    """Make AttentiveFP network.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: AttentiveFP model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None, 'input_edge_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 5, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'attention_args': {"units": 32},
                     'depth': 3,
                     'dropout': 0.1
                     }
    m = update_model_args(model_default, model_args)
    print("INFO: Updated functional make model kwargs:")
    pprint.pprint(m)

    # Local variables for model args
    input_node_shape= m['input_node_shape']
    input_edge_shape= m['input_edge_shape']
    depth = m['depth']
    dropout = m['dropout']
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    attention_args = m['attention_args']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    # Embedding, if no feature dimension
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
    edi = edge_index_input

    # Model
    nk = Dense(units=attention_args['units'])(n)
    Ck = AttentiveHeadFP(use_edge_features=True, **attention_args)([nk, ed, edi])
    nk = GRUUpdate(units=attention_args['units'])([nk, Ck])

    for i in range(1, depth):
        Ck = AttentiveHeadFP(**attention_args)([nk, ed, edi])
        nk = GRUUpdate(units=attention_args['units'])([nk, Ck])
        nk = Dropout(rate=dropout)(nk)
    n = nk

    # Output embedding choice
    if output_embedding["output_mode"] == 'graph':
        out = PoolingNodesAttentive(units=attention_args['units'])(n)
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
    return model


try:
    # Haste version of AttentiveFP
    from kgcnn.layers.haste import HasteLayerNormGRUUpdate, HastePoolingNodesAttentiveLayerNorm

    def make_haste_attentiveFP(**kwargs):
        """Make AttentiveFP network.

        Args:
            **kwargs

        Returns:
            tf.keras.models.Model: AttentiveFP model.
        """
        model_args = kwargs
        model_default = {'input_node_shape': None, 'input_edge_shape': None,
                         'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                             "edges": {"input_dim": 5, "output_dim": 64},
                                             "state": {"input_dim": 100, "output_dim": 64}},
                         'output_embedd': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                         'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                        "activation": ['relu', 'relu', 'sigmoid']},
                         'attention_args': {"units": 32},
                         'depth': 3, 'dropout': 0.1, 'verbose': 1
                         }
        m = update_model_args(model_default, model_args)
        if m['verbose'] > 0:
            print("INFO: Updated functional make model kwargs:")
            pprint.pprint(m)

        # Local variables for model args
        input_node_shape = m['input_node_shape']
        input_edge_shape = m['input_edge_shape']
        depth = m['depth']
        dropout = m['dropout']
        input_embedding = m['input_embedding']
        output_embedding = m['output_embedding']
        output_mlp = m['output_mlp']
        attention_args = m['attention_args']

        # Make input
        node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
        edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
        edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

        # Embedding, if no feature dimension
        n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
        ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
        edi = edge_index_input

        # Model
        nk = Dense(units=attention_args['units'])(n)
        Ck = AttentiveHeadFP(use_edge_features=True,**attention_args)([nk,ed,edi])
        nk = HasteLayerNormGRUUpdate(units=attention_args['units'], dropout=dropout)([nk, Ck])

        for i in range(1, depth):
            Ck = AttentiveHeadFP(**attention_args)([nk,ed,edi])
            nk = HasteLayerNormGRUUpdate(units=attention_args['units'], dropout=dropout)([nk, Ck])
        n = nk

        # Output embedding choice
        if output_embedding["output_mode"] == 'graph':
            out = HastePoolingNodesAttentiveLayerNorm(units=attention_args['units'], dropout=dropout)(n)
            output_mlp.update({"input_tensor_type": "tensor"})
            out = MLP(**output_mlp)(out)
            main_output = ks.layers.Flatten()(out)  # will be dense
        else:  # node embedding
            out = MLP(**output_mlp)(n)
            main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

        model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)
        return model
except:
    print("WARNING: Haste implementation not available.")