import keras as ks
from kgcnn.layers.message import MessagePassingBase
from kgcnn.layers.norm import GraphBatchNormalization
from keras.layers import Activation, Multiply, Concatenate, Add, Dense


class CGCNNLayer(MessagePassingBase):
    r"""Message Passing Layer used in the Crystal Graph Convolutional Neural Network:

    `CGCNN <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`__ .

    Based on the original code in pytorch (<https://github.com/txie-93/cgcnn>).

    Args:
        units (int): Units for Dense layer.
        activation_s (str): Tensorflow activation applied before gating the message.
        activation_out (str): Tensorflow activation applied the very end of the layer (after gating).
        batch_normalization (bool): Whether to use batch normalization (:obj:`GraphBatchNormalization`) or not.
        use_bias (bool): Boolean, whether the layer uses a bias vector. Default is True.
        kernel_initializer: Initializer for the `kernel` weights matrix. Default is "glorot_uniform".
        bias_initializer: Initializer for the bias vector. Default is "zeros".
        padded_disjoint: Whether disjoint tensors have padded nodes. Default if False.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix. Default is None.
        bias_regularizer: Regularizer function applied to the bias vector. Default is None.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation"). Default is None.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix. Default is None.
        bias_constraint: Constraint function applied to the bias vector. Default is None.
    """

    def __init__(self, units: int = 64,
                 activation_s="softplus",
                 activation_out="softplus",
                 batch_normalization: bool = False,
                 use_bias: bool = True,
                 kernel_regularizer=None,
                 padded_disjoint: bool = False,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 pooling_method: str = "scatter_mean",
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CGCNNLayer, self).__init__(use_id_tensors=4, pooling_method=pooling_method, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.padded_disjoint = padded_disjoint
        self.batch_normalization = batch_normalization
        kernel_args = {"kernel_regularizer": kernel_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.activation_f_layer = Activation(activation="sigmoid", activity_regularizer=activity_regularizer)
        self.activation_s_layer = Activation(activation_s, activity_regularizer=activity_regularizer)
        self.activation_out_layer = Activation(activation_out, activity_regularizer=activity_regularizer)
        if batch_normalization:
            self.batch_norm_f = GraphBatchNormalization(padded_disjoint=padded_disjoint)
            self.batch_norm_s = GraphBatchNormalization(padded_disjoint=padded_disjoint)
            self.batch_norm_out = GraphBatchNormalization(padded_disjoint=padded_disjoint)
        self.f = Dense(self.units, activation="linear", use_bias=use_bias, **kernel_args)
        self.s = Dense(self.units, activation="linear", use_bias=use_bias, **kernel_args)
        self.lazy_mult = Multiply()
        self.lazy_add = Add()
        self.lazy_concat = Concatenate(axis=-1)

    def message_function(self, inputs, **kwargs):
        r"""Prepare messages.

        Args:
            inputs: [nodes_in, nodes_out, edges, graph_id_node, graph_id_edge, count_nodes, count_edges]

                - nodes_in (Tensor): Embedding of sending nodes of shape `([M], F)`
                - nodes_out (Tensor): Embedding of sending nodes of shape `([M], F)`
                - edges (Tensor): Embedding of edges of shape `([M], E)`
                - graph_id_node (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - graph_id_edge (Tensor): ID tensor of batch assignment in disjoint graph of shape `([M], )` .
                - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
                - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .

        Returns:
            Tensor: Messages for updates of shape `([M], units)`.
        """
        nodes_in = inputs[0]  # shape: (batch_size, M, F)
        nodes_out = inputs[1]  # shape: (batch_size, M, F)
        edge_features = inputs[2]  # shape: (batch_size, M, E)
        graph_id_node, graph_id_edge, count_nodes, count_edges = inputs[3:]

        x = self.lazy_concat([nodes_in, nodes_out, edge_features], **kwargs)
        x_s, x_f = self.s(x, **kwargs), self.f(x, **kwargs)
        if self.batch_normalization:
            x_s = self.batch_norm_s([x_s, graph_id_edge, count_edges], **kwargs)
            x_f = self.batch_norm_f([x_f, graph_id_edge, count_edges], **kwargs)
        x_s, x_f = self.activation_s_layer(x_s, **kwargs), self.activation_f_layer(x_f, **kwargs)
        x_out = self.lazy_mult([x_s, x_f], **kwargs)  # shape: (batch_size, M, self.units)
        return x_out

    def update_nodes(self, inputs, **kwargs):
        """Update node embeddings.

        Args:
            inputs: [nodes, nodes_updates, graph_id_node, graph_id_edge, count_nodes, count_edges]

                - nodes (Tensor): Embedding of nodes of previous layer of shape `([M], F)`
                - nodes_updates (Tensor): Node updates of shape `([M], F)`
                - graph_id_node (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - graph_id_edge (Tensor): ID tensor of batch assignment in disjoint graph of shape `([M], )` .
                - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
                - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .

        Returns:
            Tensor: Updated nodes of shape `([N], F)`.
        """
        nodes = inputs[0]
        nodes_update = inputs[1]
        graph_id_node, graph_id_edge, count_nodes, count_edges = inputs[2:]

        if self.batch_normalization:
            nodes_update = self.batch_norm_out([nodes_update, graph_id_node, count_nodes], **kwargs)

        nodes_updated = self.lazy_add([nodes, nodes_update], **kwargs)
        nodes_updated = self.activation_out_layer(nodes_updated, **kwargs)
        return nodes_updated

    def get_config(self):
        """Update layer config."""
        config = super(CGCNNLayer, self).get_config()
        config.update({
            "units": self.units, "use_bias": self.use_bias, "padded_disjoint": self.padded_disjoint,
            "batch_normalization": self.batch_normalization})
        conf_s = self.activation_s_layer.get_config()
        conf_out = self.activation_out_layer.get_config()
        config.update({"activation_s": conf_s["activation"]})
        config.update({"activation_out": conf_out["activation"]})
        if "activity_regularizer" in conf_out.keys():
            config.update({"activity_regularizer": conf_out["activity_regularizer"]})
        conf_f = self.f.get_config()
        for x in ["kernel_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer"]:
            if x in conf_f.keys():
                config.update({x: conf_f[x]})
        return config
