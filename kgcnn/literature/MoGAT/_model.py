from kgcnn.layers.update import GRUUpdate
from keras.layers import Flatten, Add
from kgcnn.layers.modules import Embedding, ExpandDims, SqueezeDims
from kgcnn.layers.pooling import PoolingNodesAttentive
from keras.layers import Dense, Dropout, Concatenate, Attention
from kgcnn.layers.mlp import MLP, GraphMLP
from ._layers import AttentiveHeadFP_


def model_disjoint(
        inputs,
        use_node_embedding: bool = None,
        use_edge_embedding: bool = None,
        input_node_embedding: dict = None,
        input_edge_embedding: dict = None,
        attention_args=None,
        dropout=None,
        depthato=None,
        depthmol=None,
        output_embedding=None,
        output_mlp=None,
        pooling_gat_nodes_args=None
):
    # Model implementation with disjoint representation.
    n, ed, edi, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    # Model
    nk = Dense(units=attention_args['units'])(n)
    ck = AttentiveHeadFP_(use_edge_features=True, **attention_args)([nk, ed, edi])
    nk = GRUUpdate(units=attention_args['units'])([nk, ck])
    nk = Dropout(rate=dropout)(nk)  # adding dropout to the first code not in the original AttFP code ?
    list_emb = [nk]  # "aka r1"
    for i in range(1, depthato):
        ck = AttentiveHeadFP_(**attention_args)([nk, ed, edi])
        nk = GRUUpdate(units=attention_args['units'])([nk, ck])
        nk = Dropout(rate=dropout)(nk)
        list_emb.append(nk)

    # we store representation of each atomic nodes (at r1,r2,...)

    if output_embedding == 'graph':
        # we apply a super node to each atomic node representation and concate them
        out = [
            # Tensor output.
            PoolingNodesAttentive(units=attention_args['units'], depth=depthmol)([count_nodes, ni, batch_id_node]) for
            ni in list_emb
        ]
        out = [ExpandDims(axis=1)(x) for x in out]
        out = Concatenate(axis=1)(out)
        # we compute the weigthed scaled self-attention of the super nodes
        at = Attention(dropout=dropout, use_scale=True, score_mode="dot")([out, out])
        # we apply the dot product
        out = at * out
        out = Flatten()(out)
        # in the paper this is only one dense layer to the target ... very simple
        out = MLP(**output_mlp)(out)

    elif output_embedding == 'node':
        n = Add()(list_emb)
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported graph embedding for mode `MoGAT` .")

    return out
