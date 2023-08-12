import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import partition_row_indexing
from kgcnn.ops.segment import segment_ops_by_name, segment_softmax, pad_segments
from kgcnn.ops.scatter import tensor_scatter_nd_ops_by_name, supported_scatter_modes

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='AggregateLocalEdges')
class AggregateLocalEdges(GraphBaseLayer):
    r"""The main aggregation or pooling layer to collect all edges or edge-like embeddings per node,
    corresponding to the receiving node, which is defined by edge indices.

    Apply e.g. sum or mean on edges with same target ID taken from the (edge) index tensor, that has a list of
    all connections as :math:`(i, j)` . In the default definition for this layer index :math:`i` is expected ot be the
    receiving or target node (in standard case of directed edges). This can be changed by setting :obj:`pooling_index` ,
    i.e. `index_tensor[:, :, pooling_index]` to get the indices to aggregate the edges with.

    .. note::

         You can choose whether to use scatter or segment operation by 'scatter_sum' or 'segment_sum'. If leaving it up
         to the layer just specify 'sum'. Note that for segment operation, the indices should be sorted. If
         `is_sorted` is set to `False` , indices are always sorted by default which will cost performance. If
         `is_sorted` is `True` segment operation are usually most efficient. Note that some layers in
         :obj:`kgcnn.layers.aggr` , that inherit from this layer, only support segment or scatter operation for
         aggregation. Having sorted indices with `is_sorted` to `True` is therefore favourable.
    """

    def __init__(self,
                 pooling_method: str = "mean",
                 pooling_index: int = 0,
                 is_sorted: bool = False,
                 has_unconnected: bool = True,
                 **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
            pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
            is_sorted (bool): Whether node indices are sorted. Default is False.
            has_unconnected (bool): Whether graphs have unconnected nodes. Default is True.
        """
        super(AggregateLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.pooling_index = pooling_index
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = "sample"

    def build(self, input_shape):
        """Build layer."""
        super(AggregateLocalEdges, self).build(input_shape)

    @staticmethod
    def _sort_by(indices, *args):
        out = []
        indices_order = tf.argsort(indices, axis=0, direction='ASCENDING', stable=True)
        out.append(tf.gather(indices, indices_order, axis=0))
        for v in args:
            out.append(tf.gather(v, indices_order, axis=0))
        return out

    def _aggregate(self, nodes, edges, receive_indices):
        # Aggregate via scatter ops.
        if "scatter" in self.pooling_method:
            _use_scatter = True
        elif not self.is_sorted and self.pooling_method in supported_scatter_modes:
            _use_scatter = True
        else:
            _use_scatter = False

        if _use_scatter:
            return tensor_scatter_nd_ops_by_name(
                self.pooling_method,
                tf.zeros(tf.concat([tf.shape(nodes)[:1], tf.shape(edges)[1:]], axis=0), dtype=edges.dtype),
                tf.expand_dims(receive_indices, axis=-1), edges
            )
        # Aggregation via segments ops.
        if not self.is_sorted:
            receive_indices, edges = self._sort_by(receive_indices, edges)
        out = segment_ops_by_name(
            self.pooling_method, edges, receive_indices, max_id=tf.shape(nodes)[0] if self.has_unconnected else None)
        return out

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

        nodes, node_part = inputs[0].values, inputs[0].row_splits
        edges, _ = inputs[1].values, inputs[1].row_lengths()
        edge_indices, edge_part = inputs[2].values, inputs[2].row_lengths()

        disjoint_indices = partition_row_indexing(
            edge_indices, node_part, edge_part,
            partition_type_target="row_splits",
            partition_type_index="row_length",
            to_indexing='batch',
            from_indexing=self.node_indexing
        )
        receive_indices = disjoint_indices[:, self.pooling_index]  # Pick index eg. ingoing

        out = self._aggregate(nodes, edges, receive_indices)

        return tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)

    def get_config(self):
        """Update layer config."""
        config = super(AggregateLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method, "pooling_index": self.pooling_index,
                       "is_sorted": self.is_sorted, "has_unconnected": self.has_unconnected})
        return config


# Alias for compatibility.
AggregateLocalMessages = AggregateLocalEdges
PoolingLocalEdges = AggregateLocalEdges
PoolingLocalMessages = AggregateLocalEdges


@ks.utils.register_keras_serializable(package='kgcnn', name='AggregateWeightedLocalEdges')
class AggregateWeightedLocalEdges(AggregateLocalEdges):
    r"""This class inherits from :obj:`AggregateLocalEdges` for aggregating weighted edges.

    Please check the documentation of :obj:`AggregateLocalEdges` for more information.

    .. note::

        In addition to aggregating edge embeddings a weight tensor must be supplied that scales each edge before
        pooling. Must broadcast.
    """

    def __init__(self,
                 normalize_by_weights: bool = False,
                 pooling_method: str = "mean",
                 pooling_index: int = 0,
                 is_sorted: bool = False,
                 has_unconnected: bool = True,
                 **kwargs):
        """Initialize layer.

        Args:
            normalize_by_weights (bool): Whether to normalize pooled features by the sum of weights. Default is False.
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
            pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
            is_sorted (bool): Whether node indices are sorted. Default is False.
            has_unconnected (bool): Whether graphs have unconnected nodes. Default is True.
        """
        super(AggregateWeightedLocalEdges, self).__init__(
            pooling_method=pooling_method, pooling_index=pooling_index, is_sorted=is_sorted,
            has_unconnected=has_unconnected, **kwargs)
        self.normalize_by_weights = normalize_by_weights

    def build(self, input_shape):
        """Build layer."""
        super(AggregateWeightedLocalEdges, self).build(input_shape)

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
        inputs = self.assert_ragged_input_rank(inputs)
        nodes, node_part = inputs[0].values, inputs[0].row_splits
        edges, _ = inputs[1].values, inputs[1].row_lengths()
        edge_indices, edge_part = inputs[2].values, inputs[2].row_lengths()
        weights, _ = inputs[3].values, inputs[3].row_lengths()

        disjoint_indices = partition_row_indexing(
            edge_indices, node_part, edge_part,
            partition_type_target="row_splits", partition_type_index="row_length",
            to_indexing='batch', from_indexing=self.node_indexing
        )
        receive_indices = disjoint_indices[:, self.pooling_index]  # Pick index eg. ingoing
        edges_weighted = edges * weights

        out = self._aggregate(nodes, edges_weighted, receive_indices)

        if self.normalize_by_weights:
            norm = tensor_scatter_nd_ops_by_name(
                "add", tf.zeros(tf.concat([tf.shape(nodes)[:1], tf.shape(edges)[1:]], axis=0), dtype=edges.dtype),
                tf.expand_dims(receive_indices, axis=-1), weights
            )
            # We could also optionally add tf.eps here.
            out = tf.math.divide_no_nan(out, norm)

        return tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)

    def get_config(self):
        """Update layer config."""
        config = super(AggregateWeightedLocalEdges, self).get_config()
        config.update({"normalize_by_weights": self.normalize_by_weights})
        return config


# Alias for compatibility.
PoolingWeightedLocalEdges = AggregateWeightedLocalEdges
PoolingWeightedLocalMessages = AggregateWeightedLocalEdges
AggregateWeightedLocalMessages = AggregateWeightedLocalEdges


@ks.utils.register_keras_serializable(package='kgcnn', name='AggregateLocalEdgesLSTM')
class AggregateLocalEdgesLSTM(AggregateLocalEdges):
    r"""This class inherits from :obj:`AggregateLocalEdges` for aggregating edges via a LSTM.

    Please check the documentation of :obj:`AggregateLocalEdges` for more information.

    .. note::

        Apply LSTM on edges with same target ID taken from the (edge) index_tensor. Uses keras LSTM layer internally.
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
                 is_sorted: bool = False,
                 has_unconnected: bool = True,
                 **kwargs):
        """Initialize layer.

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
            is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
            has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        """
        super(AggregateLocalEdgesLSTM, self).__init__(
            pooling_method=pooling_method, pooling_index=pooling_index, is_sorted=is_sorted,
            has_unconnected=has_unconnected, **kwargs)
        self.node_indexing = "sample"
        self.lstm_unit = ks.layers.LSTM(
            units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout,
            recurrent_dropout=recurrent_dropout, return_sequences=return_sequences, return_state=return_state,
            go_backwards=go_backwards, stateful=stateful, time_major=time_major, unroll=unroll
        )
        if self.pooling_method not in ["LSTM", "lstm"]:
            raise ValueError(
                "Warning: Aggregate method does not match layer, expected 'LSTM' but got '%s'." % self.pooling_method)

    def build(self, input_shape):
        """Build layer."""
        super(AggregateLocalEdgesLSTM, self).build(input_shape)

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
        inputs = self.assert_ragged_input_rank(inputs)

        nodes, node_part = inputs[0].values, inputs[0].row_splits
        edges, _ = inputs[1].values, inputs[1].row_lengths()
        edge_indices, edge_part = inputs[2].values, inputs[2].row_lengths()

        disjoint_indices = partition_row_indexing(
            edge_indices, node_part, edge_part,
            partition_type_target="row_splits", partition_type_index="row_length",
            to_indexing='batch', from_indexing=self.node_indexing
        )

        receive_indices = disjoint_indices[:, self.pooling_index]

        if not self.is_sorted:
            receive_indices, edges = self._sort_by(receive_indices, edges)

        # aggregate via LSTM, we make a ragged input.
        ragged_lstm_input = tf.RaggedTensor.from_value_rowids(edges, receive_indices)
        out = self.lstm_unit(ragged_lstm_input)

        if self.has_unconnected:
            out = pad_segments(out, tf.shape(nodes)[0])

        return tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)

    def get_config(self):
        """Update layer config."""
        config = super(AggregateLocalEdgesLSTM, self).get_config()
        conf_lstm = self.lstm_unit.get_config()
        lstm_param = ["activation", "recurrent_activation", "use_bias", "kernel_initializer", "recurrent_initializer",
                      "bias_initializer", "unit_forget_bias", "kernel_regularizer", "recurrent_regularizer",
                      "bias_regularizer", "activity_regularizer", "kernel_constraint", "recurrent_constraint",
                      "bias_constraint", "dropout", "recurrent_dropout", "implementation", "return_sequences",
                      "return_state", "go_backwards", "stateful", "time_major", "unroll"]
        for x in lstm_param:
            config.update({x: conf_lstm[x]})
        return config


PoolingLocalMessagesLSTM = AggregateLocalEdgesLSTM  # For now, they are synonyms
PoolingLocalEdgesLSTM = AggregateLocalEdgesLSTM
AggregateLocalMessagesLSTM = AggregateLocalEdgesLSTM


@ks.utils.register_keras_serializable(package='kgcnn', name='AggregateLocalEdgesAttention')
class AggregateLocalEdgesAttention(AggregateLocalEdges):
    r"""This class inherits from :obj:`AggregateLocalEdges` for aggregating edges via attention weights.

    Please check the documentation of :obj:`AggregateLocalEdges` for more information.

    Uses attention for pooling. i.e. :math:`n_i =  \sum_j \alpha_{ij} e_{ij}`
    The attention is computed via: :math:`\alpha_ij = \text{softmax}_j (a_{ij})` from the
    attention coefficients :math:`a_{ij}` .
    The attention coefficients must be computed beforehand by edge features or by :math:`\sigma( W n_i || W n_j)` and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, :math:`n_i = \sum_j \text{softmax}_j (a_{ij}) e_{ij}` is computed by the layer.
    """

    def __init__(self,
                 pooling_index: int = 0,
                 is_sorted: bool = False,
                 has_unconnected: bool = True,
                 **kwargs):
        """Initialize layer.

        Args:
            pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
            is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
            has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        """
        super(AggregateLocalEdgesAttention, self).__init__(
            pooling_method="attention", pooling_index=pooling_index, is_sorted=is_sorted,
            has_unconnected=has_unconnected, **kwargs)
        if self.pooling_method not in ["attention"]:
            raise ValueError(
                "Warning: Aggregate method expected 'attention' but got '%s'." % self.pooling_method)

    def build(self, input_shape):
        """Build layer."""
        super(AggregateLocalEdgesAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [node, edges, attention, edge_indices]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - attention (tf.RaggedTensor): Attention coefficients of shape (batch, [M], 1)
                - edge_indices (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled edge attentions for each node of shape (batch, [N], F)
        """
        inputs = self.assert_ragged_input_rank(inputs, ragged_rank=1)

        nodes, node_part = inputs[0].values, inputs[0].row_lengths()
        edges = inputs[1].values
        attention = inputs[2].values
        edge_indices, edge_part = inputs[3].values, inputs[3].row_lengths()

        disjoint_indices = partition_row_indexing(
            edge_indices, node_part, edge_part, partition_type_target="row_length",
            partition_type_index="row_length", to_indexing='batch', from_indexing=self.node_indexing
        )

        receive_indices = disjoint_indices[:, self.pooling_index]

        if not self.is_sorted:
            receive_indices, edges, attention = self._sort_by(receive_indices, edges, attention)

        # Apply segmented softmax
        alpha = segment_softmax(attention, receive_indices)
        edges_attended = edges * alpha
        out = tf.math.segment_sum(edges_attended, receive_indices)

        if self.has_unconnected:
            out = pad_segments(out, tf.shape(nodes)[0])

        return tf.RaggedTensor.from_row_lengths(out, node_part, validate=self.ragged_validate)

    def get_config(self):
        """Update layer config."""
        config = super(AggregateLocalEdgesAttention, self).get_config()
        return config


PoolingLocalEdgesAttention = AggregateLocalEdgesAttention
AggregateLocalMessagesAttention = AggregateLocalEdgesAttention


@ks.utils.register_keras_serializable(package='kgcnn', name='RelationalAggregateLocalEdges')
class RelationalAggregateLocalEdges(AggregateLocalEdges):
    r"""This class inherits from :obj:`AggregateLocalEdges` for aggregating relational edges.

    Please check the documentation of :obj:`AggregateLocalEdges` for more information.

    The main aggregation or pooling layer to collect all edges or edge-like embeddings per node, per relation,
    corresponding to the receiving node, which is defined by edge indices.

    .. note::

        An edge relation tensor must be provided which specifies the relation for each edge.
    """

    def __init__(self, num_relations: int,
                 pooling_method: str = "sum",
                 pooling_index: int = 0,
                 is_sorted: bool = False,
                 has_unconnected: bool = True,
                 **kwargs):
        """Initialize layer.

        Args:
            num_relations (int): Number of possible relations.
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
            pooling_index (int): Index from edge_indices to pick ID's for pooling edge-like embeddings. Default is 0.
            is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
            has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        """
        super(RelationalAggregateLocalEdges, self).__init__(
            pooling_method=pooling_method, pooling_index=pooling_index, is_sorted=is_sorted,
            has_unconnected=has_unconnected, **kwargs)
        self.num_relations = num_relations

    def build(self, input_shape):
        """Build layer."""
        super(RelationalAggregateLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, tensor_index]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message features of shape (batch, [M], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)
                - edge_relation (tf.RaggedTensor): Edge relation for each edge of shape (batch, [M])

        Returns:
            tf.RaggedTensor: Pooled feature tensor of pooled edge features for each node.
        """
        inputs = self.assert_ragged_input_rank(inputs)

        nodes, node_part = inputs[0].values, inputs[0].row_splits
        edges, _ = inputs[1].values, inputs[1].row_lengths()
        edge_indices, edge_part = inputs[2].values, inputs[2].row_lengths()
        edge_relations, _ = inputs[3].values, inputs[3].row_lengths()

        disjoint_indices = partition_row_indexing(
            edge_indices, node_part, edge_part,
            partition_type_target="row_splits", partition_type_index="row_length",
            to_indexing='batch', from_indexing=self.node_indexing
        )

        receive_indices = disjoint_indices[:, self.pooling_index]  # Pick index eg. ingoing
        relations = tf.cast(edge_relations, receive_indices.dtype)
        scatter_indices = tf.concat(
            [tf.expand_dims(receive_indices, axis=-1), tf.expand_dims(relations, axis=-1)], axis=-1)
        out_tensor = tf.zeros(tf.concat(
            [tf.shape(nodes)[:1], tf.constant([self.num_relations], dtype=tf.shape(nodes).dtype),
             tf.shape(edges)[1:]], axis=0))

        # Pooling via scatter
        out = tensor_scatter_nd_ops_by_name(self.pooling_method, out_tensor, scatter_indices, edges)

        return tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)

    def get_config(self):
        """Update layer config."""
        config = super(RelationalAggregateLocalEdges, self).get_config()
        config.update({"num_relations": self.num_relations})
        return config


RelationalPoolingLocalEdges = RelationalAggregateLocalEdges
RelationalAggregateLocalMessages = RelationalAggregateLocalEdges
