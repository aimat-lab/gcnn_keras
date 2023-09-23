import keras_core as ks
# import keras_core.saving
from keras_core.layers import Layer
from keras_core import ops
from kgcnn.ops.scatter import (
    scatter_reduce_min, scatter_reduce_mean, scatter_reduce_max, scatter_reduce_sum, scatter_reduce_softmax)


@ks.saving.register_keras_serializable(package='kgcnn', name='Aggregate')
class Aggregate(Layer):  # noqa

    def __init__(self, pooling_method: str = "scatter_sum", axis=0, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.axis = axis
        if axis != 0:
            raise NotImplementedError()
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
        
    def build(self, input_shape):
        # Nothing to build here. No sub-layers.
        self.built = True
        
    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        x_shape, _, dim_size = input_shape
        return tuple(list(dim_size[:1]) + list(x_shape[1:]))

    def call(self, inputs, **kwargs):
        x, index, reference = inputs
        shape = ops.shape(reference)[:1] + ops.shape(x)[1:]
        if self._use_scatter:
            return self._pool_method(index, x, shape=shape)
        else:
            raise NotImplementedError()


class AggregateLocalEdges(Layer):

    def __init__(self, pooling_method="scatter_sum", pooling_index: int = 1, **kwargs):
        super(AggregateLocalEdges, self).__init__(**kwargs)
        self.pooling_index = pooling_index
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        assert len(input_shape) == 3
        node_shape, edges_shape, edge_index_shape = input_shape
        self.to_aggregate.build((edges_shape, edge_index_shape[1:], node_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        node_shape, edges_shape, edge_index_shape = input_shape
        return self.to_aggregate.compute_output_shape([edges_shape, edge_index_shape[1:], node_shape])

    def call(self, inputs, **kwargs):
        n, edges, edge_index = inputs
        return self.to_aggregate([edges, edge_index[self.pooling_index], n])


class AggregateWeightedLocalEdges(AggregateLocalEdges):

    def __init__(self, pooling_method="scatter_sum", pooling_index: int = 1, normalize_by_weights=False, **kwargs):
        super(AggregateWeightedLocalEdges, self).__init__(**kwargs)
        self.normalize_by_weights = normalize_by_weights
        self.pooling_index = pooling_index
        self.to_aggregate = Aggregate(pooling_method=pooling_method)
        self.to_aggregate_weights = Aggregate(pooling_method="scatter_sum")

    def build(self, input_shape):
        assert len(input_shape) == 4
        node_shape, edges_shape, edge_index_shape, weights_shape = input_shape
        self.to_aggregate.build((edges_shape, edge_index_shape[1:], node_shape))
        self.to_aggregate_weights.build((weights_shape, edge_index_shape[1:], node_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        node_shape, edges_shape, edge_index_shape, weights_shape = input_shape
        return self.to_aggregate.compute_output_shape([edges_shape, edge_index_shape[1:], node_shape])

    def call(self, inputs, **kwargs):
        n, edges, edge_index, weights = inputs
        edges = edges*weights

        out = self.to_aggregate([edges, edge_index[self.pooling_index], n])

        if self.normalize_by_weights:
            norm = self.to_aggregate_weights([weights, edge_index[self.pooling_index], n])
            out = out/norm
        return out


class AggregateLocalEdgesAttention(Layer):
    r"""Aggregate local edges via Attention mechanism.
    Uses attention for pooling. i.e. :math:`n_i =  \sum_j \alpha_{ij} e_{ij}`
    The attention is computed via: :math:`\alpha_ij = \text{softmax}_j (a_{ij})` from the
    attention coefficients :math:`a_{ij}` .
    The attention coefficients must be computed beforehand by edge features or by :math:`\sigma( W n_i || W n_j)` and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, :math:`n_i = \sum_j \text{softmax}_j (a_{ij}) e_{ij}` is computed by the layer.
    """

    def __init__(self,
                 softmax_method="scatter_softmax",
                 pooling_method="scatter_sum",
                 pooling_index: int = 1,
                 is_sorted: bool = False,
                 has_unconnected: bool = True,
                 normalize: bool = False,
                 **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method for this layer.
            pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
            is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
            has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        """
        super(AggregateLocalEdgesAttention, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.pooling_index = pooling_index
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.normalize = normalize
        self.softmax_method = softmax_method
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        assert len(input_shape) == 4
        node_shape, edges_shape, attention_shape, edge_index_shape = input_shape
        self.to_aggregate.build((edges_shape, edge_index_shape[1:], node_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        node_shape, edges_shape, attention_shape, edge_index_shape = input_shape
        return self.to_aggregate.compute_output_shape([edges_shape, edge_index_shape[1:], node_shape])

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [node, edges, attention, edge_indices]

                - nodes (Tensor): Node embeddings of shape (N, F)
                - edges (Tensor): Edge or message embeddings of shape (M, F)
                - attention (Tensor): Attention coefficients of shape (M, 1)
                - edge_indices (Tensor): Edge indices referring to nodes of shape (2, M)

        Returns:
            Tensor: Embedding tensor of aggregated edge attentions for each node of shape (N, F)
        """
        reference, x, attention, disjoint_indices = inputs
        receive_indices = disjoint_indices[self.pooling_index]
        shape_attention = ops.shape(reference)[:1] + ops.shape(attention)[1:]
        a = scatter_reduce_softmax(receive_indices, attention, shape=shape_attention, normalize=self.normalize)
        x = x * ops.broadcast_to(a, ops.shape(x))
        return self.to_aggregate([x, receive_indices, reference])

    def get_config(self):
        """Update layer config."""
        config = super(AggregateLocalEdgesAttention, self).get_config()
        return config


class AggregateLocalEdgesLSTM(Layer):

    def __init__(self,
                 units: int,
                 max_edges_per_node: int,
                 pooling_method="LSTM",
                 pooling_index=1,
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
            pooling_method (str): Pooling method. Default is 'LSTM', is ignored.
            pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 1.
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

        n, edges, disjoint_indices, edge_id = inputs
        receive_indices = disjoint_indices[self.pooling_index]

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
        return config
