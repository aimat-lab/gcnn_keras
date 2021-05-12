import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type, kgcnn_ops_check_tensor_type


class TrafoMatMulMessages(ks.layers.Layer):
    """
    Apply message by edge matrix multiplication.
    
    The message dimension must be suitable for matrix multiplication.
    
    Args:
        target_shape (int): Target dimension. Message dimension must match target_dim*node_dim.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self, target_shape,
                 node_indexing="sample",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(TrafoMatMulMessages, self).__init__(**kwargs)
        self.target_shape = target_shape
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(TrafoMatMulMessages, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge,
        i.e. (batch, None, 2)

        Args:
            inputs (list): of [trafo, edges]

            - trafo: Transformation by matrix multiplication for each message. Must be reshaped to (batch, [N], FxF).
            - edges: Edge embeddings or messages (batch, [N], F)
            
        Returns:
            node_updates: Transformation of messages by matrix multiplication of shape (batch, [N], F)
        """
        found_trafo_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                       node_indexing=self.node_indexing)
        found_edge_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        dens_trafo, trafo_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_trafo_type,
                                                    output_tensor_type="values_partition",
                                                    partition_type=self.partition_type)
        dens_e, epart = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_edge_type,
                                           output_tensor_type="values_partition",
                                           partition_type=self.partition_type)

        dens_m = tf.reshape(dens_trafo,
                            (ks.backend.shape(dens_trafo)[0], self.target_shape, ks.backend.shape(dens_e)[-1]))
        out = tf.keras.backend.batch_dot(dens_m, dens_e)

        return kgcnn_ops_dyn_cast([out, epart], input_tensor_type="values_partition",
                                  output_tensor_type=found_edge_type, partition_type=self.partition_type)

    def get_config(self):
        """Update layer config."""
        config = super(TrafoMatMulMessages, self).get_config()
        config.update({"target_shape": self.target_shape,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class GRUupdate(ks.layers.Layer):
    """
    Gated recurrent unit update.
    
    Args:
        units (int): Units for GRU.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `glorot_uniform`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent state.
            Default: `orthogonal`.
        bias_initializer: Initializer for the bias vector. Default: `zeros`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector. Default:
            `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel`
            weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector. Default:
            `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
            the linear transformation of the recurrent state. Default: 0.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before",
            True = "after" (default and CuDNN compatible).
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self, units,
                 activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros', kernel_regularizer=None,
                 recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                 recurrent_dropout=0.0, reset_after=True,
                 node_indexing="sample",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(GRUupdate, self).__init__(**kwargs)
        self.units = units
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

        self.gru_cell = tf.keras.layers.GRUCell(units=units,
                                                activation=activation, recurrent_activation=recurrent_activation,
                                                use_bias=use_bias, kernel_initializer=kernel_initializer,
                                                recurrent_initializer=recurrent_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                recurrent_regularizer=recurrent_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                recurrent_constraint=recurrent_constraint,
                                                bias_constraint=bias_constraint,
                                                dropout=dropout,
                                                recurrent_dropout=recurrent_dropout, reset_after=reset_after)

    def build(self, input_shape):
        """Build layer."""
        # self.gru.build(channels)
        super(GRUupdate, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge,
        i.e. (batch, None, 2)

        Args:
            inputs (list): of [nodes, updates]

            - nodes (tf.tensor): Node embeddings of shape (batch, [N], F)
            - updates (tf.tensor): Matching node updates of shape (batch, [N], F

        Returns:
            updated_nodes (tf.tensor): Updated nodes of shape (batch*None,F)
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_updates_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                         node_indexing=self.node_indexing)
        n, npart = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                      output_tensor_type="values_partition",
                                      partition_type=self.partition_type)
        eu, _ = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_updates_type,
                                   output_tensor_type="values_partition",
                                   partition_type=self.partition_type)

        out, _ = self.gru_cell(eu, [n], **kwargs)

        return kgcnn_ops_dyn_cast([out, npart], input_tensor_type="values_partition",
                                  output_tensor_type=found_node_type, partition_type=self.partition_type)

    def get_config(self):
        """Update layer config."""
        config = super(GRUupdate, self).get_config()
        conf_cell = self.gru_cell.get_config()
        config.update({"is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        param_list = ["units", "activation", "recurrent_activation",
                      "use_bias", "kernel_initializer",
                      "recurrent_initializer",
                      "bias_initializer", "kernel_regularizer",
                      "recurrent_regularizer", "bias_regularizer", "kernel_constraint",
                      "recurrent_constraint", "bias_constraint", "dropout",
                      "recurrent_dropout", "reset_after"]
        for x in param_list:
            config.update({x: conf_cell[x]})
        return config
