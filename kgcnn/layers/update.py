import tensorflow as tf
import tensorflow.keras as ks


class ApplyMessage(ks.layers.Layer):
    """
    Apply message by edge matrix multiplication.
    
    The message dimension must be suitable for matrix multiplication.
    
    Args:
        target_shape (int): Target dimension. Message dimension must match target_dim*node_dim.
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
        super(ApplyMessage, self).__init__(**kwargs)
        self.target_shape = target_shape
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._tensor_input_type_implemented = ["ragged", "values_partition"]
        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)

    def build(self, input_shape):
        """Build layer."""
        super(ApplyMessage, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [message, nodes]

            - message (tf.tensor): Message tensor of shape (batch*None,target_dim*node_dim)
              that can be reshaped to (batch*None,target_shape,node_dim)
            - nodes (tf.tensor): Node feature list of shape (batch*None,node_dim)
            
        Returns:
            node_updates (tf.tensor): Element-wise matmul of message and node features
            of output shape (batch,target_dim)
        """
        dens_e, dens_n = inputs
        dens_m = tf.reshape(dens_e, (ks.backend.shape(dens_e)[0], self.target_shape, ks.backend.shape(dens_n)[-1]))
        out = tf.keras.backend.batch_dot(dens_m, dens_n)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(ApplyMessage, self).get_config()
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
        self._tensor_input_type_implemented = ["ragged", "values_partition"]
        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)

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

        Args:
            inputs (list): of [nodes, updates]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - updates (tf.tensor): Matching node updates of same shape (batch*None,F)
            
        Returns:
            updated_nodes (tf.tensor): Updated nodes of shape (batch*None,F)
        """
        if self.input_tensor_type=="values_partition":
            [n, npart], [eu, _] = inputs
            # Apply GRU for update node state
            out, _ = self.gru_cell(eu, [n])
            return [out, npart]
        elif self.input_tensor_type=="values_partition":
            n, eu = inputs
            out, _ = self.gru_cell(eu.values, [n.values])
            return tf.RaggedTensor.from_row_lengths(out, n.row_lengths())

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
