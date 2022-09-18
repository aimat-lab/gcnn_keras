import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import partition_row_indexing
from kgcnn.ops.segment import segment_ops_by_name, segment_softmax

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingLocalEdges')
class PoolingLocalEdges(GraphBaseLayer):
    r"""The main aggregation or pooling layer to collect all edges or edge-like embeddings per node,
    corresponding to the receiving node, which is defined by edge indices.
    The term pooling is here used as aggregating rather than reducing the graph as in graph pooling.

    Apply e.g. sum or mean on edges with same target ID taken from the (edge) index tensor, that has a list of
    all connections as :math:`(i, j)`. In the default definition for this layer index :math:`i` is expected ot be the
    receiving or target node (in standard case of directed edges). This can be changed by setting :obj:`pooling_index`.

    Note: index_tensor[:, :, pooling_index] is sorted for the subsequent segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        pooling_index (int): Index from edge_indices to pick ID's for pooling edge-like embeddings. Default is 0.
    """

    def __init__(self, pooling_method="mean", pooling_index=0, **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.pooling_index = pooling_index

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, tensor_index]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message features of shape (batch, [M], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)
    
        Returns:
            tf.RaggedTensor: Pooled feature tensor of pooled edge features for each node.
        """
        # Need ragged input but can be generalized in the future.
        self.assert_ragged_input_rank(inputs)

        nod, node_part = inputs[0].values, inputs[0].row_splits
        edge, _ = inputs[1].values, inputs[1].row_lengths()
        edgeind, edge_part = inputs[2].values, inputs[2].row_lengths()

        shiftind = partition_row_indexing(edgeind, node_part, edge_part,
                                          partition_type_target="row_splits",
                                          partition_type_index="row_length",
                                          to_indexing='batch',
                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, self.pooling_index]  # Pick index eg. ingoing
        dens = edge
        if not self.is_sorted:
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)

        # Pooling via e.g. segment_sum
        out = segment_ops_by_name(self.pooling_method, dens, nodind)

        # If not unsort_segment operation need a scatter here.
        if self.has_unconnected:
            out = tf.scatter_nd(ks.backend.expand_dims(tf.range(tf.shape(out)[0]), axis=-1), out,
                                tf.concat([tf.shape(nod)[:1], tf.shape(out)[1:]], axis=0))

        out = tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method, "pooling_index": self.pooling_index})
        return config


PoolingLocalMessages = PoolingLocalEdges  # For now, they are synonyms


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingWeightedLocalEdges')
class PoolingWeightedLocalEdges(GraphBaseLayer):
    r"""The main aggregation or pooling layer to collect all edges or edge-like embeddings per node,
    corresponding to the receiving node, which is defined by edge indices.
    The term pooling is here used as aggregating rather than reducing the graph as in graph pooling.

    Apply e.g. sum or mean on edges with same target ID taken from the (edge) index-tensor, that has a list of
    all connections as :math:`(i, j)`. In the default definition for this layer index :math:`i` is expected ot be the
    receiving or target node (in standard case of directed edges). This can be changed by setting :obj:`pooling_index`.

    .. note::
        In addition of aggregating edge embeddings a weight tensor must be supplied that scales each edge before
        pooling. Must broadcast.

    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
    """

    def __init__(self, pooling_method="mean",
                 normalize_by_weights=False,
                 pooling_index=0,
                 **kwargs):
        """Initialize layer."""
        super(PoolingWeightedLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.normalize_by_weights = normalize_by_weights
        self.pooling_index = pooling_index

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, tensor_index, weights]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message features of shape (batch, [M], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)
                - weights (tf.RaggedTensor): Edge or message weights. Must broadcast to edges or messages,
                  e.g. (batch, [M], 1)

        Returns:
            tf.RaggedTensor: Pooled feature tensor of pooled edge features for each node of shape (batch, [N], F)
        """
        self.assert_ragged_input_rank(inputs)

        nod, node_part = inputs[0].values, inputs[0].row_splits
        edge, _ = inputs[1].values, inputs[1].row_lengths()
        edgeind, edge_part = inputs[2].values, inputs[2].row_lengths()
        weights, _ = inputs[3].values, inputs[3].row_lengths()

        shiftind = partition_row_indexing(edgeind, node_part, edge_part,
                                          partition_type_target="row_splits", partition_type_index="row_length",
                                          to_indexing='batch', from_indexing=self.node_indexing)

        wval = weights
        dens = edge * wval
        nodind = shiftind[:, self.pooling_index]

        if not self.is_sorted:
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)
            wval = tf.gather(wval, node_order, axis=0)

        # Pooling via e.g. segment_sum
        get = segment_ops_by_name(self.pooling_method, dens, nodind)

        if self.normalize_by_weights:
            get = tf.math.divide_no_nan(get, tf.math.segment_sum(wval, nodind))  # +tf.eps

        if self.has_unconnected:
            get = tf.scatter_nd(ks.backend.expand_dims(tf.range(tf.shape(get)[0]), axis=-1), get,
                                tf.concat([tf.shape(nod)[:1], tf.shape(get)[1:]], axis=0))

        out = tf.RaggedTensor.from_row_splits(get, node_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method, "normalize_by_weights": self.normalize_by_weights,
                       "pooling_index": self.pooling_index})
        return config


PoolingWeightedLocalMessages = PoolingWeightedLocalEdges  # For now, they are synonyms


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingEmbedding')
class PoolingEmbedding(GraphBaseLayer):
    """Polling all embeddings of edges or nodes per batch to obtain a graph level embedding in form of a
    ::obj`tf.Tensor`.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
    """

    def __init__(self, pooling_method="mean", **kwargs):
        """Initialize layer."""
        super(PoolingEmbedding, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Embedding tensor of shape (batch, [N], F)
    
        Returns:
            tf.Tensor: Pooled node features of shape (batch, F)
        """
        # Need ragged input but can be generalized in the future.
        self.assert_ragged_input_rank(inputs)
        # We cast to values here
        nod, batchi = inputs.values, inputs.value_rowids()

        # Could also use reduce_sum here.
        out = segment_ops_by_name(self.pooling_method, nod, batchi)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingEmbedding, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


PoolingNodes = PoolingEmbedding
PoolingGlobalEdges = PoolingEmbedding


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingWeightedEmbedding')
class PoolingWeightedEmbedding(GraphBaseLayer):
    """Polling all embeddings of edges or nodes per batch to obtain a graph level embedding in form of a
    ::obj`tf.Tensor`.

    .. note::
        In addition of pooling embeddings a weight tensor must be supplied that scales each embedding before
        pooling. Must broadcast.

    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
    """

    def __init__(self, pooling_method="mean", **kwargs):
        """Initialize layer."""
        super(PoolingWeightedEmbedding, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, weights]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - weights (tf.RaggedTensor): Node or message weights. Most broadcast to nodes. Shape (batch, [N], 1).

        Returns:
            tf.Tensor: Pooled node features of shape (batch, F)
        """
        # Need ragged input but can be generalized in the future.
        self.assert_ragged_input_rank(inputs)
        # We cast to values here
        nod, batchi = inputs[0].values, inputs[0].value_rowids()
        weights, _ = inputs[1].values, inputs[1].value_rowids()
        nod = tf.math.multiply(nod, weights)
        # Could also use reduce_sum here.
        out = segment_ops_by_name(self.pooling_method, nod, batchi)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedEmbedding, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


PoolingWeightedNodes = PoolingWeightedEmbedding
PoolingWeightedGlobalEdges = PoolingWeightedEmbedding


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingLocalEdgesLSTM')
class PoolingLocalEdgesLSTM(GraphBaseLayer):
    """The main aggregation or pooling layer to collect all edges or edge-like embeddings per node,
    corresponding to the receiving node, which is defined by edge indices.
    The term pooling is here used as aggregating rather than reducing the graph as in graph pooling.

    Here, apply LSTM on edges with same target ID taken from the (edge) index_tensor, that has a list of
    all connections as :math:`(i, j)`. In the default definition for this layer index :math:`i` is expected ot be the
    receiving or target node (in standard case of directed edges). This can be changed by setting :obj:`pooling_index`.

    Args:
        units (int): Units for LSTM cell.
        pooling_method (str): Pooling method. Default is 'LSTM', is ignored.
        pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
            is applied (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean (default `True`), whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix, used for
            the linear transformation of the inputs. Default: `glorot_uniform`.
            recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
            Default: `orthogonal`.
        bias_initializer: Initializer for the bias vector. Default: `zeros`.
            unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
            the forget gate at initialization. Setting it to true will also force
            `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
            bias_regularizer: Regularizer function applied to the bias vector. Default:
            `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel`
            weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector. Default:
            `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear
            transformation of the inputs. Default: 0.
            recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
            the linear transformation of the recurrent state. Default: 0.
        return_sequences: Boolean. Whether to return the last output. in the output
            sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition to the
            output. Default: `False`.
        go_backwards: Boolean (default `False`). If True, process the input sequence
            backwards and return the reversed sequence.
        stateful: Boolean (default `False`). If True, the last state for each sample
            at index i in a batch will be used as initial state for the sample of
            index i in the following batch.
        time_major: The shape format of the `inputs` and `outputs` tensors.
            If True, the inputs and outputs will be in shape
            `[timesteps, batch, feature]`, whereas in the False case, it will be
            `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
            efficient because it avoids transposes at the beginning and end of the
            RNN calculation. However, most TensorFlow data is batch-major, so by
            default this function accepts input and emits output in batch-major
            form.
        unroll: Boolean (default `False`). If True, the network will be unrolled,
            else a symbolic loop will be used. Unrolling can speed-up a RNN, although
            it tends to be more memory-intensive. Unrolling is only suitable for short
            sequences.
    """

    def __init__(self,
                 units,
                 pooling_method="LSTM",
                 pooling_index=0,
                 activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros', unit_forget_bias=True,
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                 bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=False, return_state=False, go_backwards=False, stateful=False,
                 time_major=False, unroll=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdgesLSTM, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.pooling_index = pooling_index

        self.lstm_unit = ks.layers.LSTM(units=units, activation=activation, recurrent_activation=recurrent_activation,
                                        use_bias=use_bias, kernel_initializer=kernel_initializer,
                                        recurrent_initializer=recurrent_initializer,
                                        bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias,
                                        kernel_regularizer=kernel_regularizer,
                                        recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                                        recurrent_constraint=recurrent_constraint,
                                        bias_constraint=bias_constraint, dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        return_sequences=return_sequences, return_state=return_state,
                                        go_backwards=go_backwards, stateful=stateful,
                                        time_major=time_major, unroll=unroll)
        if self.pooling_method not in ["LSTM", "lstm"]:
            print("Warning: Pooling method does not match with layer, expected 'LSTM' but got", self.pooling_method)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdgesLSTM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, edges, tensor_index]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message features of shape (batch, [M], F)
                - tensor_index (tf.RaggedTensor): Edge indices of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Feature tensor of pooled edge features for each node of shape (batch, [N], F)
        """
        self.assert_ragged_input_rank(inputs)

        nod, node_part = inputs[0].values, inputs[0].row_splits
        edge, _ = inputs[1].values, inputs[1].row_lengths()
        edgeind, edge_part = inputs[2].values, inputs[2].row_lengths()

        shiftind = partition_row_indexing(edgeind, node_part, edge_part,
                                          partition_type_target="row_splits",
                                          partition_type_index="row_length",
                                          to_indexing='batch',
                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, self.pooling_index]  # Pick first index eg. ingoing
        dens = edge
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)

        # Pooling via LSTM
        # we make a ragged input
        ragged_lstm_input = tf.RaggedTensor.from_value_rowids(dens, nodind)
        get = self.lstm_unit(ragged_lstm_input)

        if self.has_unconnected:
            # Need to fill tensor since the maximum node may not be also in pooled
            # Does not happen if all nodes are also connected
            get = tf.scatter_nd(ks.backend.expand_dims(tf.range(tf.shape(get)[0]), axis=-1), get,
                                tf.concat([tf.shape(nod)[:1], tf.shape(get)[1:]], axis=0))

        out = tf.RaggedTensor.from_row_splits(get, node_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesLSTM, self).get_config()
        config.update({"pooling_method": self.pooling_method, "pooling_index": self.pooling_index})
        conf_lstm = self.lstm_unit.get_config()
        lstm_param = ["activation", "recurrent_activation", "use_bias", "kernel_initializer", "recurrent_initializer",
                      "bias_initializer", "unit_forget_bias", "kernel_regularizer", "recurrent_regularizer",
                      "bias_regularizer", "activity_regularizer", "kernel_constraint", "recurrent_constraint",
                      "bias_constraint", "dropout", "recurrent_dropout", "implementation", "return_sequences",
                      "return_state", "go_backwards", "stateful", "time_major", "unroll"]
        for x in lstm_param:
            config.update({x: conf_lstm[x]})
        return config


PoolingLocalMessagesLSTM = PoolingLocalEdgesLSTM  # For now they are synonyms


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingLocalEdgesAttention')
class PoolingLocalEdgesAttention(GraphBaseLayer):
    r"""Pooling or aggregation of all edges or edge-like features per node,
    corresponding to node assigned by edge indices.
    Uses attention for pooling. i.e. :math:`n_i =  \sum_j \alpha_{ij} e_{ij}`
    The attention is computed via: :math:`\alpha_ij = \text{softmax}_j (a_{ij})` from the
    attention coefficients :math:`a_{ij}`.
    The attention coefficients must be computed beforehand by edge features or by :math:`\sigma( W n_i || W n_j)` and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, :math:`n_i = \sum_j \text{softmax}_j (a_{ij}) e_{ij}` is computed by the layer.

    An edge is defined by index tuple :math:`(i, j)` with :math:`i` being the receiving node in the default definition,
    but can be changed by pooling_index.
    Important: ID's for segment-operation and for pooling of edges are taken from edge-index-tensor.
    They are sorted for faster pooling from tensor_index[:, :, pooling_index].

    Args:
        pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
    """

    def __init__(self, pooling_index=0, **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdgesAttention, self).__init__(**kwargs)
        self.pooling_index = pooling_index

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdgesAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [node, edges, attention, edge_indices]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - attention (tf.RaggedTensor): Attention coefficients of shape (batch, [M], 1)
                - edge_indices (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], F)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled edge attentions for each node of shape (batch, [N], F)
        """
        # Need ragged input but can be generalized in the future.
        self.assert_ragged_input_rank(inputs)
        # We cast to values here
        nod, node_part = inputs[0].values, inputs[0].row_lengths()
        edge = inputs[1].values
        attention = inputs[2].values
        edgeind, edge_part = inputs[3].values, inputs[3].row_lengths()

        shiftind = partition_row_indexing(edgeind, node_part, edge_part, partition_type_target="row_length",
                                          partition_type_index="row_length", to_indexing='batch',
                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, self.pooling_index]  # Pick first index eg. ingoing
        dens = edge
        ats = attention
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)
            ats = tf.gather(ats, node_order, axis=0)

        # Apply segmented softmax
        ats = segment_softmax(ats, nodind)
        get = dens * ats
        get = tf.math.segment_sum(get, nodind)

        if self.has_unconnected:
            # Need to fill tensor since the maximum node may not be also in pooled
            # Does not happen if all nodes are also connected
            get = tf.scatter_nd(ks.backend.expand_dims(tf.range(tf.shape(get)[0]), axis=-1), get,
                                tf.concat([tf.shape(nod)[:1], tf.shape(get)[1:]], axis=0))

        out = tf.RaggedTensor.from_row_lengths(get, node_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesAttention, self).get_config()
        config.update({"pooling_index": self.pooling_index})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingEmbeddingAttention')
class PoolingEmbeddingAttention(GraphBaseLayer):
    r"""Polling all embeddings of edges or nodes per batch to obtain a graph level embedding in form of a
    ::obj`tf.Tensor`.

    Uses attention for pooling. i.e. :math:`s =  \sum_j \alpha_{i} n_i`.
    The attention is computed via: :math:`\alpha_i = \text{softmax}_i(a_i)` from the attention coefficients :math:`a_i`.
    The attention coefficients must be computed beforehand by edge features or by :math:`\sigma( W [s || n_i])` and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, :math:`s =  \sum_i \text{softmax}_j(a_i) n_i` is computed by the layer.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(PoolingEmbeddingAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingEmbeddingAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, attention]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - attention (tf.RaggedTensor): Attention coefficients of shape (batch, [N], 1)

        Returns:
            tf.Tensor: Embedding tensor of pooled node of shape (batch, F)
        """
        # Need ragged input but can be generalized in the future.
        self.assert_ragged_input_rank(inputs)
        # We cast to values here
        nod, batchi, target_len = inputs[0].values, inputs[0].value_rowids(), inputs[0].row_lengths()
        ats = inputs[1].values

        ats = segment_softmax(ats, batchi)
        get = nod * ats
        out = tf.math.segment_sum(get, batchi)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingEmbeddingAttention, self).get_config()
        return config


PoolingNodesAttention = PoolingEmbeddingAttention
