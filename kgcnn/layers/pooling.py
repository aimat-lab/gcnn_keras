import keras as ks
from keras.layers import Layer, Dense, Concatenate, GRUCell, Activation
from kgcnn.layers.gather import GatherState
from keras import ops
import kgcnn.ops.activ
from kgcnn.ops.scatter import scatter_reduce_softmax
from kgcnn.layers.aggr import Aggregate


class PoolingNodes(Layer):
    r"""Main layer to pool node or edge attributes. Uses :obj:`Aggregate` layer."""

    def __init__(self, pooling_method="scatter_sum", **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'scatter_sum'.
        """
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self._to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        """Build Layer."""
        self._to_aggregate.build([input_shape[1], input_shape[2], input_shape[0]])
        self.built = True

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return self._to_aggregate.compute_output_shape([input_shape[1], input_shape[2], input_shape[0]])

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [reference, attr, weights, batch_index]

                - reference (Tensor): Reference for aggregation of shape `(batch, ...)` .
                - attr (Tensor): Node or edge embeddings of shape `([N], F)` .
                - batch_index (Tensor): Batch assignment of shape `([N], )` .

        Returns:
            Tensor: Embedding tensor of pooled node of shape `(batch, F)` .
        """
        reference, x, idx = inputs
        return self._to_aggregate([x, idx, reference])

    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


class PoolingWeightedNodes(Layer):
    r"""Weighted polling all embeddings of edges or nodes per batch to obtain a graph level embedding.

    .. note::

        In addition to pooling embeddings a weight tensor must be supplied that scales each embedding before
        pooling. Must broadcast.
    """

    def __init__(self, pooling_method="scatter_sum", **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'scatter_sum'.
        """
        super(PoolingWeightedNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self._to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        """Build layer."""
        assert len(input_shape) == 4
        ref_shape, attr_shape, weights_shape, index_shape = [list(x) for x in input_shape]
        self._to_aggregate.build([tuple(x) for x in [attr_shape, index_shape, ref_shape]])
        self.built = True

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [reference, attr, weights, batch_index]

                - reference (Tensor): Reference for aggregation of shape `(batch, ...)` .
                - attr (Tensor): Node or edge embeddings of shape `([N], F)` .
                - weights (Tensor): Node or message weights. Most broadcast to nodes. Shape ([N], 1).
                - batch_index (Tensor): Batch assignment of shape `([N], )` .

        Returns:
            Tensor: Embedding tensor of pooled node of shape `(batch, F)` .
        """
        # Need ragged input but can be generalized in the future.
        reference, x, w, idx = inputs
        xw = ops.broadcast_to(ops.cast(w, dtype=x.dtype), ops.shape(x)) * x
        return self._to_aggregate([xw, idx, reference])

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


class PoolingEmbeddingAttention(Layer):
    r"""Polling all embeddings of edges or nodes per batch to obtain a graph level embedding in form of a
    :obj:`Tensor` .

    Uses attention for pooling. i.e. :math:`s =  \sum_j \alpha_{i} n_i` .
    The attention is computed via: :math:`\alpha_i = \text{softmax}_i(a_i)` from the attention
    coefficients :math:`a_i` .
    The attention coefficients must be computed beforehand by node or edge features or by :math:`\sigma( W [s || n_i])`
    and are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, :math:`s =  \sum_i \text{softmax}_j(a_i) n_i` is computed by the layer.
    """

    def __init__(self,
                 softmax_method="scatter_softmax",
                 pooling_method="scatter_sum",
                 normalize_softmax: bool = False,
                 **kwargs):
        """Initialize layer.

        Args:
            normalize_softmax (bool): Whether to use normalize in softmax. Default is False.
        """
        super(PoolingEmbeddingAttention, self).__init__(**kwargs)
        self.normalize_softmax = normalize_softmax
        self.pooling_method = pooling_method
        self.softmax_method = softmax_method
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        """Build layer."""
        assert len(input_shape) == 4
        ref_shape, attr_shape, attention_shape, index_shape = [list(x) for x in input_shape]
        self.to_aggregate.build([tuple(x) for x in [attr_shape, index_shape, ref_shape]])
        self.built = True

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [reference, attr, attention, batch_index]

                - reference (Tensor): Reference for aggregation of shape `(batch, ...)` .
                - attr (Tensor): Node or edge embeddings of shape `([N], F)` .
                - attention (Tensor): Attention coefficients of shape `([N], 1)` .
                - batch_index (Tensor): Batch assignment of shape `([N], )` .

        Returns:
            Tensor: Embedding tensor of pooled node of shape `(batch, F)` .
        """
        reference, attr, attention, batch_index = inputs
        shape_attention = ops.shape(reference)[:1] + ops.shape(attention)[1:]
        a = scatter_reduce_softmax(batch_index, attention, shape=shape_attention, normalize=self.normalize_softmax)
        x = attr * ops.broadcast_to(a, ops.shape(attr))
        return self.to_aggregate([x, batch_index, reference])

    def get_config(self):
        """Update layer config."""
        config = super(PoolingEmbeddingAttention, self).get_config()
        config.update({
            "normalize_softmax": self.normalize_softmax, "pooling_method": self.pooling_method,
            "softmax_method": self.softmax_method
        })
        return config


PoolingNodesAttention = PoolingEmbeddingAttention


class PoolingNodesAttentive(Layer):
    r"""Computes the attentive pooling for node embeddings for
    `Attentive FP <https://doi.org/10.1021/acs.jmedchem.9b00959>`__ model.
    """

    def __init__(self,
                 units,
                 depth=3,
                 pooling_method="sum",
                 activation="kgcnn>leaky_relu2",
                 activation_context="elu",
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 recurrent_activation='sigmoid',
                 recurrent_initializer='orthogonal',
                 recurrent_regularizer=None,
                 recurrent_constraint=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 reset_after=True,
                 **kwargs):
        """Initialize layer.
        
        Args:
            units (int): Units for the linear trafo of node features before attention.
            pooling_method(str): Initial pooling before iteration. Default is "sum".
            depth (int): Number of iterations for graph embedding. Default is 3.
            activation (str): Activation. Default is "kgcnn>leaky_relu2".
            activation_context (str): Activation function for context. Default is "elu".
            use_bias (bool): Use bias. Default is True.
            kernel_regularizer: Kernel regularization. Default is None.
            bias_regularizer: Bias regularization. Default is None.
            activity_regularizer: Activity regularization. Default is None.
            kernel_constraint: Kernel constrains. Default is None.
            bias_constraint: Bias constrains. Default is None.
            kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
            bias_initializer: Initializer for bias. Default is 'zeros'.
        """
        super(PoolingNodesAttentive, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
        self.pooling_method = pooling_method
        self.depth = depth
        self.units = int(units)
        kernel_args = {"use_bias": use_bias, "kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        gru_args = {"recurrent_activation": recurrent_activation,
                    "use_bias": use_bias, "kernel_initializer": kernel_initializer,
                    "recurrent_initializer": recurrent_initializer, "bias_initializer": bias_initializer,
                    "kernel_regularizer": kernel_regularizer, "recurrent_regularizer": recurrent_regularizer,
                    "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                    "recurrent_constraint": recurrent_constraint, "bias_constraint": bias_constraint,
                    "dropout": dropout, "recurrent_dropout": recurrent_dropout, "reset_after": reset_after}

        self.lay_linear_trafo = Dense(units, activation="linear", **kernel_args)
        self.lay_alpha = Dense(1, activation=activation, **kernel_args)
        self.lay_gather_s = GatherState()
        self.lay_concat = Concatenate(axis=-1)
        self.lay_pool_start = PoolingNodes(pooling_method=self.pooling_method)
        self.lay_pool_attention = PoolingNodesAttention()
        self.lay_final_activ = Activation(activation=activation_context)
        self.lay_gru = GRUCell(units=units, activation="tanh", **gru_args)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodesAttentive, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [reference, nodes, batch_index]

                - reference (Tensor): Reference for aggregation of shape `(batch, ...)` .
                - nodes (Tensor): Node embeddings of shape `([N], F)` .
                - batch_index (Tensor): Batch assignment of shape `([N], )` .

        Returns:
            Tensor: Hidden tensor of pooled node attentions of shape (batch, F).
        """
        ref, node, batch_index = inputs

        h = self.lay_pool_start([ref, node, batch_index], **kwargs)
        wn = self.lay_linear_trafo(node, **kwargs)
        for _ in range(self.depth):
            hv = self.lay_gather_s([h, batch_index], **kwargs)
            ev = self.lay_concat([hv, node], **kwargs)
            av = self.lay_alpha(ev, **kwargs)
            cont = self.lay_pool_attention([ref, wn, av, batch_index], **kwargs)
            cont = self.lay_final_activ(cont, **kwargs)
            h, _ = self.lay_gru(cont, h, **kwargs)

        out = h
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodesAttentive, self).get_config()
        config.update({"units": self.units, "depth": self.depth, "pooling_method": self.pooling_method})
        conf_sub = self.lay_alpha.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            if x in conf_sub.keys():
                config.update({x: conf_sub[x]})
        conf_context = self.lay_final_activ.get_config()
        config.update({"activation_context": conf_context["activation"]})
        conf_gru = self.lay_gru.get_config()
        for x in ["recurrent_activation", "recurrent_initializer", "recurrent_regularizer", "recurrent_constraint",
                  "dropout", "recurrent_dropout", "reset_after"]:
            if x in conf_gru.keys():
                config.update({x: conf_gru[x]})
        return config
