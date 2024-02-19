import keras as ks
# import keras_core.saving
from keras.layers import Layer
from keras import ops
from kgcnn.ops.scatter import (
    scatter_reduce_min, scatter_reduce_mean, scatter_reduce_max, scatter_reduce_sum, scatter_reduce_softmax)
from kgcnn import __indices_axis__ as global_axis_indices
from kgcnn import __index_receive__ as global_index_receive


@ks.saving.register_keras_serializable(package='kgcnn', name='Aggregate')
class Aggregate(Layer):  # noqa
    """Main class for aggregating node or edge features.

    The class essentially uses a reduce function by name to aggregate a feature list given indices to group by.
    Possible supported permutation invariant aggregations are 'sum', 'mean', 'max' or 'min'.
    For aggregation either scatter or segment operation can be used from the backend, if available.
    Note that you have to specify which to use with e.g. 'scatter_sum'.
    This layer further requires a reference tensor to either statically infer the output shape or even directly
    aggregate the values into.
    """

    def __init__(self, pooling_method: str = "scatter_sum", axis=0, **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Method for aggregation. Default is 'scatter_sum'.
            axis (int): Axis to aggregate. Default is 0.
        """
        super(Aggregate, self).__init__(**kwargs)
        # Shorthand notation
        if pooling_method == "sum":
            pooling_method = "scatter_sum"
        if pooling_method == "max":
            pooling_method = "scatter_max"
        if pooling_method == "min":
            pooling_method = "scatter_min"
        if pooling_method == "mean":
            pooling_method = "scatter_mean"
        self.pooling_method = pooling_method
        self.axis = axis
        if axis != 0:
            raise NotImplementedError("Only aggregate at axis=0 is supported for `Aggregate` layer.")
        pooling_by_name = {
            "scatter_sum": scatter_reduce_sum,
            "scatter_mean": scatter_reduce_mean,
            "scatter_max": scatter_reduce_max,
            "scatter_min": scatter_reduce_min,
            "segment_sum": None,
            "segment_mean": None,
            "segment_max": None,
            "segment_min": None
        }
        self._pool_method = pooling_by_name[pooling_method]
        self._use_scatter = "scatter" in pooling_method
        self._use_reference_for_aggregation = "update" in pooling_method

    def build(self, input_shape):
        """Build layer."""
        # Nothing to build here. No sub-layers.
        self.built = True

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        assert len(input_shape) == 3
        x_shape, _, dim_size = input_shape
        return tuple(list(dim_size[:1]) + list(x_shape[1:]))

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [values, indices, reference]

                - values (Tensor): Values to aggregate of shape `(M, ...)`.
                - indices (Tensor): Indices of target assignment of shape `(M, )`.
                - reference (Tensor): Target reference tensor of shape `(N, ...)`.

        Returns:
            Tensor: Aggregated values of shape `(N, ...)`.
        """
        x, index, reference = inputs
        shape = ops.shape(reference)[:1] + ops.shape(x)[1:]
        if self._use_scatter:
            return self._pool_method(index, x, shape=shape)
        else:
            raise NotImplementedError()

    def get_config(self):
        """Get config for layer."""
        conf = super(Aggregate, self).get_config()
        conf.update({"pooling_method": self.pooling_method, "axis": self.axis})
        return conf


@ks.saving.register_keras_serializable(package='kgcnn', name='AggregateLocalEdges')
class AggregateLocalEdges(Layer):
    r"""The main aggregation or pooling layer to collect all edges or edge-like embeddings per node,
    corresponding to the receiving node, which is defined by edge indices.

    Apply e.g. 'sum' or 'mean' on edges with same target ID taken from the (edge) index tensor, that has a list of
    all connections as :math:`(i, j)` .

    In the default definition for this layer index :math:`i` is expected to be the
    receiving or target node (in standard case of directed edges). This can be changed by setting :obj:`pooling_index` ,
    i.e. `index_tensor[pooling_index]` to get the indices to aggregate the edges with.
    This layers uses the :obj:`Aggregate` layer and its functionality.
    """
    def __init__(self,
                 pooling_method="scatter_sum",
                 pooling_index: int = global_index_receive,
                 axis_indices: int = global_axis_indices,
                 **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'scatter_sum'.
            pooling_index (int): Index to pick IDs for pooling edge-like embeddings. Default is 0.
            axis_indices (bool): The axis of the index tensor to pick IDs from. Default is 0.
        """
        super(AggregateLocalEdges, self).__init__(**kwargs)
        self.pooling_index = pooling_index
        self.pooling_method = pooling_method
        self.to_aggregate = Aggregate(pooling_method=pooling_method)
        self.axis_indices = axis_indices

    def build(self, input_shape):
        """Build layer."""
        # Layer has no variables but still can call build on sub-layers.
        assert len(input_shape) == 3
        node_shape, edges_shape, edge_index_shape = [list(x) for x in input_shape]
        edge_index_shape.pop(self.axis_indices)
        self.to_aggregate.build([tuple(x) for x in [edges_shape, edge_index_shape, node_shape]])
        self.built = True

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        assert len(input_shape) == 3
        node_shape, edges_shape, edge_index_shape = [list(x) for x in input_shape]
        edge_index_shape.pop(self.axis_indices)
        return self.to_aggregate.compute_output_shape([tuple(x) for x in [edges_shape, edge_index_shape, node_shape]])

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [reference, values, indices]

                - reference (Tensor): Target reference tensor of shape `(N, ...)`.
                - values (Tensor): Values to aggregate of shape `(M, ...)`.
                - indices (Tensor): Indices of edges of shape `(2, M, )`.

        Returns:
            Tensor: Aggregated values of shape `(N, ...)`.
        """
        n, edges, edge_index = inputs
        receive_indices = ops.take(edge_index, self.pooling_index, axis=self.axis_indices)
        return self.to_aggregate([edges, receive_indices, n])

    def get_config(self):
        """Update layer config."""
        conf = super(AggregateLocalEdges, self).get_config()
        conf.update({"pooling_index": self.pooling_index, "pooling_method": self.pooling_method,
                     "axis_indices": self.axis_indices})
        return conf


@ks.saving.register_keras_serializable(package='kgcnn', name='AggregateWeightedLocalEdges')
class AggregateWeightedLocalEdges(AggregateLocalEdges):
    r"""This class inherits from :obj:`AggregateLocalEdges` for aggregating weighted edges.

    Please check the documentation of :obj:`AggregateLocalEdges` for more information.

    .. note::

        In addition to aggregating edge embeddings a weight tensor must be supplied that scales each edge before
        pooling. Must broadcast.
    """

    def __init__(self, pooling_method: str = "scatter_sum", pooling_index: int = global_index_receive,
                 normalize_by_weights: bool = False, axis_indices: int = global_axis_indices, **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'scatter_sum'.
            normalize_by_weights (bool): Whether to normalize pooled features by the sum of weights. Default is False.
            pooling_index (int): Index to pick IDs for pooling edge-like embeddings. Default is 0.
            axis_indices (bool): The axis of the index tensor to pick IDs from. Default is 0.
        """
        super(AggregateWeightedLocalEdges, self).__init__(**kwargs)
        self.normalize_by_weights = normalize_by_weights
        self.pooling_index = pooling_index
        self.pooling_method = pooling_method
        # to_aggregate already made by super
        if self.normalize_by_weights:
            self.to_aggregate_weights = Aggregate(pooling_method="scatter_sum")
        self.axis_indices = axis_indices

    def build(self, input_shape):
        """Build layer."""
        assert len(input_shape) == 4
        node_shape, edges_shape, edge_index_shape, weights_shape = [list(x) for x in input_shape]
        edge_index_shape.pop(self.axis_indices)
        self.to_aggregate.build([tuple(x) for x in [edges_shape, edge_index_shape, node_shape]])
        if self.normalize_by_weights:
            self.to_aggregate_weights.build([tuple(x) for x in [weights_shape, edge_index_shape, node_shape]])
        self.built = True

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        assert len(input_shape) == 4
        node_shape, edges_shape, edge_index_shape, weights_shape = [list(x) for x in input_shape]
        edge_index_shape.pop(self.axis_indices)
        return self.to_aggregate.compute_output_shape([tuple(x) for x in [edges_shape, edge_index_shape, node_shape]])

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [reference, values, indices, weights]

                - reference (Tensor): Target reference tensor of shape `(N, ...)`.
                - values (Tensor): Values to aggregate of shape `(M, ...)`.
                - indices (Tensor): Indices of edges of shape `(2, M, )`.
                - weights (Tensor): Weight tensor for values of shape `(M, ...)`.

        Returns:
            Tensor: Aggregated values of shape `(N, ...)`.
        """
        n, edges, edge_index, weights = inputs
        edges = edges*weights
        receive_indices = ops.take(edge_index, self.pooling_index, axis=self.axis_indices)
        out = self.to_aggregate([edges, receive_indices, n])

        if self.normalize_by_weights:
            norm = self.to_aggregate_weights([weights, receive_indices, n])
            out = out/norm
        return out

    def get_config(self):
        """Update layer config."""
        conf = super(AggregateWeightedLocalEdges, self).get_config()
        conf.update({"pooling_index": self.pooling_index, "pooling_method": self.pooling_method,
                     "axis_indices": self.axis_indices, "normalize_by_weights": self.normalize_by_weights})
        return conf


@ks.saving.register_keras_serializable(package='kgcnn', name='AggregateLocalEdgesAttention')
class AggregateLocalEdgesAttention(Layer):
    r"""Aggregate local edges via Attention mechanism.
    Uses attention for pooling. i.e. :math:`n_i =  \sum_j \alpha_{ij} e_{ij}`
    The attention is computed via: :math:`\alpha_ij = \text{softmax}_j (a_{ij})` from the
    attention coefficients :math:`a_{ij}` .

    The attention coefficients must be computed beforehand by edge features or by :math:`\sigma( W n_i || W n_j)` and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, the following is computed by the layer:

    .. math::

            n_i = \sum_j \text{softmax}_j (a_{ij}) e_{ij}
    """

    def __init__(self,
                 softmax_method="scatter_softmax",
                 pooling_method="scatter_sum",
                 pooling_index: int = global_index_receive,
                 is_sorted: bool = False,
                 has_unconnected: bool = True,
                 normalize_softmax: bool = False,
                 axis_indices: int = global_axis_indices,
                 **kwargs):
        """Initialize layer.

        Args:
            softmax_method (str): Method to apply softmax to attention coefficients. Default is 'scatter_softmax'.
            pooling_method (str): Pooling method for this layer. Default is 'scatter_sum'.
            pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
            is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
            has_unconnected (bool): If unconnected nodes are allowed. Default is True.
            normalize_softmax (bool): Whether to use normalize in softmax. Default is False.
            axis_indices (int): The axis of the index tensor to pick IDs from. Default is 0.
        """
        super(AggregateLocalEdgesAttention, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.pooling_index = pooling_index
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.normalize_softmax = normalize_softmax
        self.softmax_method = softmax_method
        self.to_aggregate = Aggregate(pooling_method=pooling_method)
        self.axis_indices = axis_indices

    def build(self, input_shape):
        """Build layer."""
        assert len(input_shape) == 4
        node_shape, edges_shape, attention_shape, edge_index_shape = [list(x) for x in input_shape]
        edge_index_shape.pop(self.axis_indices)
        self.to_aggregate.build([tuple(x) for x in [edges_shape, edge_index_shape, node_shape]])
        self.built = True

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        assert len(input_shape) == 4
        node_shape, edges_shape, attention_shape, edge_index_shape = [list(x) for x in input_shape]
        edge_index_shape.pop(self.axis_indices)
        return self.to_aggregate.compute_output_shape([tuple(x) for x in [edges_shape, edge_index_shape, node_shape]])

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [node, edges, attention, edge_indices]

                - nodes (Tensor): Node embeddings of shape `(N, F)`
                - edges (Tensor): Edge or message embeddings of shape `(M, F)`
                - attention (Tensor): Attention coefficients of shape `(M, 1)`
                - edge_indices (Tensor): Edge indices referring to nodes of shape `(2, M)`

        Returns:
            Tensor: Embedding tensor of aggregated edge attentions for each node of shape `(N, F)` .
        """
        reference, x, attention, edge_index = inputs
        receive_indices = ops.take(edge_index, self.pooling_index, axis=self.axis_indices)
        shape_attention = ops.shape(reference)[:1] + ops.shape(attention)[1:]
        a = scatter_reduce_softmax(receive_indices, attention, shape=shape_attention, normalize=self.normalize_softmax)
        x = x * ops.broadcast_to(a, ops.shape(x))
        return self.to_aggregate([x, receive_indices, reference])

    def get_config(self):
        """Update layer config."""
        config = super(AggregateLocalEdgesAttention, self).get_config()
        config.update({
            "normalize_softmax": self.normalize_softmax, "pooling_method": self.pooling_method,
            "pooling_index": self.pooling_index, "axis_indices": self.axis_indices,
            "softmax_method": self.softmax_method, "is_sorted": self.is_sorted, "has_unconnected": self.has_unconnected
        })
        return config


@ks.saving.register_keras_serializable(package='kgcnn', name='AggregateLocalEdgesLSTM')
class AggregateLocalEdgesLSTM(Layer):
    r"""Aggregating edges via a LSTM.

    Apply LSTM on edges with same target ID taken from the (edge) index tensor. Uses keras LSTM layer internally.

    .. note::

        Must provide a max length of edges per nodes, since keras LSTM requires padded input. Also required for use
        in connection with jax backend.
    """

    def __init__(self,
                 units: int,
                 max_edges_per_node: int,
                 pooling_method="LSTM",
                 pooling_index=global_index_receive,
                 axis_indices: int = global_axis_indices,
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
        """Initialize layer.

        Args:
            units (int): Units for LSTM cell.
            max_edges_per_node (int): Max number of edges per node.
            pooling_method (str): Pooling method. Default is 'LSTM'.
            pooling_index (int): Index to pick IDs for pooling edge-like embeddings. Default is 0.
            axis_indices (int): Axis to pick receiving index from. Default is 0.
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
        super(AggregateLocalEdgesLSTM, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.axis_indices = axis_indices
        self.pooling_index = pooling_index
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
                "Aggregate method does not match layer, expected 'LSTM' but got '%s'." % self.pooling_method)
        self.max_edges_per_node = max_edges_per_node

    def build(self, input_shape):
        """Build layer."""
        super(AggregateLocalEdgesLSTM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [node, edges, edge_indices, graph_id_edge]

                - nodes (Tensor): Node embeddings of shape `(N, F)`
                - edges (Tensor): Edge or message embeddings of shape `(M, F)`
                - edge_indices (Tensor): Edge indices referring to nodes of shape `(2, M)`
                - graph_id_edge (Tensor): Graph ID for each edge of shape `(M, )`

        Returns:
            Tensor: Embedding tensor of aggregated edges for each node of shape `(N, F)` .
        """
        n, edges, edge_index, edge_id = inputs
        receive_indices = ops.take(edge_index, self.pooling_index, axis=self.axis_indices)

        dim_n = ops.shape(n)[0]
        dim_e_per_n = self.max_edges_per_node if self.max_edges_per_node is not None else 2*dim_n+1
        indices = receive_indices * ops.convert_to_tensor(dim_e_per_n, dtype=receive_indices.dtype) + ops.cast(
            edge_id, dtype=receive_indices.dtype)
        lstm_input = scatter_reduce_sum(
            indices, edges,
            shape=tuple([dim_n*dim_e_per_n] + list(ops.shape(edges)[1:]))
        )
        lstm_mask = ops.cast(scatter_reduce_sum(
            indices, ops.ones(ops.shape(edges)[:1], dtype=ks.backend.floatx()),
            shape=tuple([dim_n*dim_e_per_n])
        ), dtype="bool")

        lstm_input = ops.reshape(lstm_input, tuple([dim_n, dim_e_per_n] + list(ops.shape(edges)[1:])))
        lstm_mask = ops.reshape(lstm_mask, tuple([dim_n, dim_e_per_n]))

        out = self.lstm_unit(lstm_input, mask=lstm_mask)
        return out

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
            if x in conf_lstm:
                config.update({x: conf_lstm[x]})
        config.update({"pooling_method": self.pooling_method, "axis_indices": self.axis_indices,
                       "pooling_index": self.pooling_index, "max_edges_per_node": self.max_edges_per_node})
        return config


@ks.saving.register_keras_serializable(package='kgcnn', name='RelationalAggregateLocalEdges')
class RelationalAggregateLocalEdges(Layer):
    r"""Layer :obj:`RelationalAggregateLocalEdges` for aggregating relational edges.

    Please check the documentation of :obj:`AggregateLocalEdges` for more information.

    The main aggregation or pooling layer to collect all edges or edge-like embeddings per node, per relation,
    corresponding to the receiving node, which is defined by edge indices.

    .. note::

        An edge relation tensor must be provided which specifies the relation for each edge.
    """

    def __init__(self,
                 num_relations: int,
                 pooling_method="scatter_sum",
                 pooling_index: int = global_index_receive,
                 axis_indices: int = global_axis_indices,
                 **kwargs):
        """Initialize layer.

        Args:
            num_relations (int): Number of possible relations.
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
            pooling_index (int): Index from edge_indices to pick ID's for pooling edge-like embeddings. Default is 0.
            axis_indices (bool): The axis of the index tensor to pick IDs from. Default is 0.
        """
        super(RelationalAggregateLocalEdges, self).__init__(**kwargs)
        self.num_relations = num_relations
        self.pooling_index = pooling_index
        self.pooling_method = pooling_method
        self.axis_indices = axis_indices
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        """Build layer."""
        super(RelationalAggregateLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): of [node, edges, tensor_index, edge_relation]

                - node (Tensor): Node reference of shape ([N], R, F)
                - edges (Tensor): Edge or message features of shape ([M], F)
                - tensor_index (Tensor): Edge indices referring to nodes of shape (2, [M])
                - edge_relation (Tensor): Edge relation for each edge of shape ([M], )

        Returns:
            Tensor: Aggregated feature tensor of edge features for each node of shape ([N], R, F) .
        """
        nodes, edges, disjoint_indices, edge_relations = inputs
        node_shape = list(ops.shape(nodes))

        receive_indices = ops.take(disjoint_indices, self.pooling_index, axis=self.axis_indices)
        relations = ops.cast(edge_relations, receive_indices.dtype)

        if self.to_aggregate._use_reference_for_aggregation:
            out_tensor = ops.reshape(nodes, [node_shape[0]*node_shape[1]] + node_shape[2:])
            shifts = node_shape[1]
        else:
            out_tensor = ops.repeat(nodes, self.num_relations, axis=0)
            shifts = self.num_relations

        scatter_indices = receive_indices*shifts + relations

        out = self.to_aggregate([edges, scatter_indices, out_tensor])
        out_shape = list(ops.shape(out))

        return ops.reshape(out, [node_shape[0], shifts] + out_shape[1:])

    def get_config(self):
        """Update layer config."""
        config = super(RelationalAggregateLocalEdges, self).get_config()
        config.update({"num_relations": self.num_relations})
        config.update({"pooling_index": self.pooling_index, "pooling_method": self.pooling_method,
                       "axis_indices": self.axis_indices})
        return config
