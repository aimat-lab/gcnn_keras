import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition
from kgcnn.ops.scatter import kgcnn_ops_scatter_segment_tensor_nd
from kgcnn.ops.segment import kgcnn_ops_segment_operation_by_name


class PoolingLocalEdges(GraphBaseLayer):
    """Pooling all edges or edge-like features per node, corresponding to node assigned by edge indices.
    
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] are sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
    """

    def __init__(self, pooling_method="mean", **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_index]

            - nodes (tf.ragged): Node features of shape (batch, [N], F)
            - edges (tf.ragged): Edge or message features of shape (batch, [M], F)
            - edge_index (tf.ragged): Edge indices of shape (batch, [M], 2)
    
        Returns:
            features: Pooled feature tensor of pooled edge features for each node.
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 3)
        # We cast to values here
        nod, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge, _ = dyn_inputs[1].values, dyn_inputs[1].row_lengths()
        edgeind, edge_part = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                          partition_type_node="row_splits",
                                                                          partition_type_edge="row_length",
                                                                          to_indexing='batch',
                                                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, 0]  # Pick first index eg. ingoing
        dens = edge
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)
        # Pooling via e.g. segment_sum
        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, dens, nodind)
        if self.has_unconnected:
            out = kgcnn_ops_scatter_segment_tensor_nd(out, nodind, tf.shape(nod))

        out = self._kgcnn_map_output_ragged([out, node_part], "row_splits", 0)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


PoolingLocalMessages = PoolingLocalEdges  # For now they are synonyms


class PoolingWeightedLocalEdges(GraphBaseLayer):
    """
    Pooling all edges or message/edge-like features per node, corresponding to node assigned by edge_indices.
    
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] could be sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
    """

    def __init__(self, pooling_method="mean",
                 normalize_by_weights=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingWeightedLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.normalize_by_weights = normalize_by_weights

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_index, weights]

            - nodes (tf.ragged): Node features of shape (batch, [N], F)
            - edges (tf.ragged): Edge or message features of shape (batch, [M], F)
            - edge_index (tf.ragged): Edge indices of shape (batch, [M], 2)
            - weights (tf.ragged): Edge or message weights. Must broadcast to edges or messages, e.g. (batch, [M], 1)

        Returns:
            features: Pooled feature tensor of pooled edge features for each node of shape (batch, [N], F)
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 4)
        # We cast to values here
        nod, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge, _ = dyn_inputs[1].values, dyn_inputs[1].row_lengths()
        edgeind, edge_part = dyn_inputs[2].values, dyn_inputs[2].row_lengths()
        weights, _ = dyn_inputs[3].values, dyn_inputs[3].row_lengths()

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                          partition_type_node="row_splits",
                                                                          partition_type_edge="row_length",
                                                                          to_indexing='batch',
                                                                          from_indexing=self.node_indexing)

        wval = weights
        dens = edge * wval
        nodind = shiftind[:, 0]

        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)
            wval = tf.gather(wval, node_order, axis=0)

        # Pooling via e.g. segment_sum
        get = kgcnn_ops_segment_operation_by_name(self.pooling_method, dens, nodind)

        if self.normalize_by_weights:
            get = tf.math.divide_no_nan(get, tf.math.segment_sum(wval, nodind))  # +tf.eps

        if self.has_unconnected:
            get = kgcnn_ops_scatter_segment_tensor_nd(get, nodind, tf.shape(nod))

        out = self._kgcnn_map_output_ragged([get, node_part], "row_splits", 0)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method, "normalize_by_weights": self.normalize_by_weights})
        return config


PoolingWeightedLocalMessages = PoolingWeightedLocalEdges  # For now they are synonyms


class PoolingNodes(GraphBaseLayer):
    """
    Polling all nodes per batch. The batch assignment is given by a length-tensor.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
    """

    def __init__(self, pooling_method="mean", **kwargs):
        """Initialize layer."""
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.ragged): Node features of shape (batch, [N], F)
    
        Returns:
            nodes (tf.tensor): Pooled node features of shape (batch, F)
        """
        dyn_inputs = self._kgcnn_map_input_ragged([inputs], 1)
        # We cast to values here
        nod, batchi = dyn_inputs[0].values, dyn_inputs[0].value_rowids()

        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, nod, batchi)

        # Output should have correct shape
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


class PoolingWeightedNodes(GraphBaseLayer):
    """
    Polling all nodes per batch. The batch assignment is given by a length-tensor.

    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
    """

    def __init__(self, pooling_method="mean", **kwargs):
        """Initialize layer."""
        super(PoolingWeightedNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, weights]

            - nodes (tf.ragged): Node features of shape (batch, [N], F)
            - weights (tf.ragged): Node or message weights. Most broadcast to nodes. Shape (batch, [N], 1).

        Returns:
            nodes (tf.tensor): Pooled node features of shape (batch,F)
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 2)
        # We cast to values here
        nod, batchi = dyn_inputs[0].values, dyn_inputs[0].value_rowids()
        weights, _ = dyn_inputs[1].values, dyn_inputs[1].value_rowids()

        nod = tf.math.multiply(nod, weights)
        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, nod, batchi)
        # Output should have correct shape
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


class PoolingGlobalEdges(GraphBaseLayer):
    """
    Pooling all edges per graph. The batch assignment is given by a length-tensor.

    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
    """

    def __init__(self, pooling_method="mean", **kwargs):
        """Initialize layer."""
        super(PoolingGlobalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingGlobalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.ragged): Edge features or message embeddings of shape (batch, [M], F)
    
        Returns:
            tf.tensor: Pooled edges feature list of shape (batch, F).
        """
        dyn_inputs = self._kgcnn_map_input_ragged([inputs], 1)
        # We cast to values here
        edge, batchi = dyn_inputs[0].values, dyn_inputs[0].value_rowids()

        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, edge, batchi)
        # Output already has correct shape and type
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingGlobalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


class PoolingLocalEdgesLSTM(GraphBaseLayer):
    """
    Pooling all edges or edge-like features per node, corresponding to node assigned by edge indices.
    Uses LSTM to aggregate Node-features.

    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    Apply e.g. segment_mean for index[0] incoming nodes.
    Important: edge_index[:,0] are sorted for segment-operation.

    Args:
        units (int): Units for LSTM cell.
        pooling_method (str): Pooling method. Default is 'LSTM', is ignored.
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
            inputs (list): [nodes, edges, edge_index]

            - nodes (tf.ragged): Node features of shape (batch, [N], F)
            - edges (tf.ragged): Edge or message features of shape (batch, [M], F)
            - edge_index (tf.ragged): Edge indices of shape (batch, [M], 2)


        Returns:
            features: Feature tensor of pooled edge features for each node of shape (batch, [N], F)
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 3)
        # We cast to values here
        nod, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits
        edge, _ = dyn_inputs[1].values, dyn_inputs[1].row_lengths()
        edgeind, edge_part = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                          partition_type_node="row_splits",
                                                                          partition_type_edge="row_length",
                                                                          to_indexing='batch',
                                                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, 0]  # Pick first index eg. ingoing
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
            get = kgcnn_ops_scatter_segment_tensor_nd(get, nodind, tf.shape(nod))

        out = self._kgcnn_map_output_ragged([get, node_part], "row_splits", 0)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesLSTM, self).get_config()
        config.update({"pooling_method": self.pooling_method})
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
