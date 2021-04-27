import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as ksb

from kgcnn.ops.activ import kgcnn_custom_act

class Dense(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 ragged_validate=False,
                 input_tensor_type = "ragged",
                 **kwargs):
        """Initialize layer same as Dense."""
        super(Dense, self).__init__(**kwargs)
        self._dense = ks.layers.Dense(units=units, activation=activation,
                                            use_bias=use_bias)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True

    def call(self, inputs, **kwargs):
        if self.input_tensor_type== "ragged":
            value_tensor = inputs.values
            out_tensor = self._dense(value_tensor)
            return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
        elif self.input_tensor_type== "values_partition":
            out = self._dense(inputs[0])
            return [out,inputs[1]]


class Activation(tf.keras.layers.Layer):

    def __init__(self, activation,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as Activation."""
        super(Activation, self).__init__(**kwargs)
        self.activation = ks.activations.get(activation)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True

    def call(self, inputs, **kwargs):
        if self.input_tensor_type == "ragged":
            out = tf.ragged.map_flat_values(self.activation, inputs)
            return out
        elif self.input_tensor_type == "values_partition":
            out = self.activation(inputs[0])
            return [out,inputs[1]]

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape

    def get_config(self):
        """Update config."""
        base_config = super(Activation, self).get_config()
        config = {'activation': ks.activations.serialize(self.activation)}
        config.update(base_config)
        return config


class Add(tf.keras.layers.Layer):

    def __init__(self,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as Activation."""
        super(Add, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._lay_add = ks.layers.Add()

    def call(self, inputs, **kwargs):
        if self.input_tensor_type == "ragged":
            out = self._lay_add(inputs)
            return out
        elif self.input_tensor_type == "values_partition":
            out_part = inputs[0][1]
            out = self._lay_add([x[0] for x in inputs])
            return [out, out_part]


class Multiply(tf.keras.layers.Layer):

    def __init__(self,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as Activation."""
        super(Multiply, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._lay_mult = ks.layers.Multiply()

    def call(self, inputs, **kwargs):
        if self.input_tensor_type == "ragged":
            out = self._lay_mult(inputs)
            return out
        elif self.input_tensor_type == "values_partition":
            out_part = inputs[0][1]
            out = self._lay_mult([x[0] for x in inputs])
            return [out, out_part]

class Concatenate(tf.keras.layers.Layer):

    def __init__(self,
                 axis,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as Activation."""
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._lay_concat = ks.layers.Concatenate(axis=self.axis)

    def call(self, inputs, **kwargs):
        if self.input_tensor_type == "ragged":
            out = self._lay_concat(inputs)
            return out
        elif self.input_tensor_type == "values_partition":
            out_part = inputs[0][1]
            out =  self._lay_concat([x[0] for x in inputs])
            return [out, out_part]