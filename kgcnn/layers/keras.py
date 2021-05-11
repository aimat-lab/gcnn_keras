import tensorflow as tf
import tensorflow.keras as ks


class Dense(tf.keras.layers.Layer):
    """Dense Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

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
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer as tf.keras.Dense."""
        super(Dense, self).__init__(**kwargs)
        self._layer_keras = ks.layers.Dense(units=units, activation=activation,
                                            use_bias=use_bias, kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tf.RaggedTensor):
            if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
            value_tensor = inputs.values
            out_tensor = self._layer_keras(value_tensor, **kwargs)
            return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
        elif isinstance(inputs, list):
            if self.input_tensor_type not in ["disjoint", "values_partition"]:
                print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
            out = self._layer_keras(inputs[0], **kwargs)
            return [out] + inputs[1:]
        elif isinstance(inputs, tf.Tensor):
            if self.input_tensor_type not in ["Tensor", "tensor"]:
                print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
            return self._layer_keras(inputs, **kwargs)
        else:
            raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        """Get config from Dense sub-layer."""
        config = super(Dense, self).get_config()
        sub_args = self._layer_keras.get_config()
        sub_args_list = ["units", "activation", "use_bias", "kernel_initializer", "bias_initializer",
                         "kernel_regularizer", "bias_regularizer", "activity_regularizer", "kernel_constraint",
                         "bias_constraint"]
        for x in sub_args_list:
            config.update({x: sub_args[x]})
        config.update({"input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class Activation(tf.keras.layers.Layer):
    """Activation Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 activation,
                 activity_regularizer=None,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as tf.keras.Activation."""
        super(Activation, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._layer_keras = tf.keras.layers.Activation(activation=activation, activity_regularizer=activity_regularizer)

    def call(self, inputs, **kwargs):
        """Forward pass."""
        if isinstance(inputs, tf.RaggedTensor):
            if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
            value_tensor = inputs.values
            out_tensor = self._layer_keras(value_tensor, **kwargs)
            return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
        elif isinstance(inputs, list):
            if self.input_tensor_type not in ["disjoint", "values_partition"]:
                print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
            out = self._layer_keras(inputs[0], **kwargs)
            return [out] + inputs[1:]
        elif isinstance(inputs, tf.Tensor):
            if self.input_tensor_type not in ["Tensor", "tensor"]:
                print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
            return self._layer_keras(inputs, **kwargs)
        else:
            raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        """Update config."""
        config = super(Activation, self).get_config()
        sub_args = self._layer_keras.get_config()
        sub_args_list = ["activation", "activity_regularizer"]
        for x in sub_args_list:
            config.update({x: sub_args[x]})
        config.update({"input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class Add(tf.keras.layers.Layer):
    """Add Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as tf.keras.Add."""
        super(Add, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._layer_keras = ks.layers.Add()

    def call(self, inputs, **kwargs):
        """Forward pass."""
        if isinstance(inputs[0], tf.RaggedTensor):
            if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
            # Works already with RaggedTensor but slower
            # out = self._layer_keras(inputs, **kwargs)
            out = self._layer_keras([x.values for x in inputs], **kwargs)
            out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
            return out
        elif isinstance(inputs[0], list):
            if self.input_tensor_type not in ["disjoint", "values_partition"]:
                print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
            out_part = inputs[0][1:]
            out = self._layer_keras([x[0] for x in inputs], **kwargs)
            return [out] + out_part
        elif isinstance(inputs[0], tf.Tensor):
            if self.input_tensor_type not in ["Tensor", "tensor"]:
                print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
            return self._layer_keras(inputs, **kwargs)
        else:
            raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        """Update config."""
        config = super(Add, self).get_config()
        config.update({"input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class Average(tf.keras.layers.Layer):
    """Average Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as tf.keras.Average."""
        super(Average, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._layer_keras = ks.layers.Average()

    def call(self, inputs, **kwargs):
        """Forward pass."""
        if isinstance(inputs[0], tf.RaggedTensor):
            if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
            # Works already with RaggedTensor but slower
            # out = self._layer_keras(inputs, **kwargs)
            out = self._layer_keras([x.values for x in inputs], **kwargs)
            out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
            return out
        elif isinstance(inputs[0], list):
            if self.input_tensor_type not in ["disjoint", "values_partition"]:
                print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
            out_part = inputs[0][1:]
            out = self._layer_keras([x[0] for x in inputs], **kwargs)
            return [out] + out_part
        elif isinstance(inputs[0], tf.Tensor):
            if self.input_tensor_type not in ["Tensor", "tensor"]:
                print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
            return self._layer_keras(inputs, **kwargs)
        else:
            raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        """Update config."""
        config = super(Average, self).get_config()
        config.update({"input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class Multiply(tf.keras.layers.Layer):
    """Multiply Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as tf.keras.Multiply."""
        super(Multiply, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._layer_keras = ks.layers.Multiply()

    def call(self, inputs, **kwargs):
        """Forward pass."""
        if isinstance(inputs[0], tf.RaggedTensor):
            if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
            # Works already with RaggedTensor but slower
            # out = self._layer_keras(inputs, **kwargs)
            out = self._layer_keras([x.values for x in inputs], **kwargs)
            out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
            return out
        elif isinstance(inputs[0], list):
            if self.input_tensor_type not in ["disjoint", "values_partition"]:
                print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
            out_part = inputs[0][1:]
            out = self._layer_keras([x[0] for x in inputs], **kwargs)
            return [out] + out_part
        elif isinstance(inputs[0], tf.Tensor):
            if self.input_tensor_type not in ["Tensor", "tensor"]:
                print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
            return self._layer_keras(inputs, **kwargs)
        else:
            raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        """Update config."""
        config = super(Multiply, self).get_config()
        config.update({"input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class Concatenate(tf.keras.layers.Layer):
    """Concatenate Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 axis,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as tf.keras.Concatenate."""
        super(Concatenate, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._layer_keras = ks.layers.Concatenate(axis=axis)

    def call(self, inputs, **kwargs):
        """Forward pass."""
        if isinstance(inputs[0], tf.RaggedTensor):
            if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
            # Works already with RaggedTensor but slower
            # out = self._layer_keras(inputs, **kwargs)
            out = self._layer_keras([x.values for x in inputs], **kwargs)
            out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
            return out
        elif isinstance(inputs[0], list):
            if self.input_tensor_type not in ["disjoint", "values_partition"]:
                print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
            out_part = inputs[0][1:]
            out = self._layer_keras([x[0] for x in inputs], **kwargs)
            return [out] + out_part
        elif isinstance(inputs[0], tf.Tensor):
            if self.input_tensor_type not in ["Tensor", "tensor"]:
                print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
            return self._layer_keras(inputs, **kwargs)
        else:
            raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        """Update config."""
        config = super(Concatenate, self).get_config()
        sub_args = self._layer_keras.get_config()
        sub_args_list = ["axis"]
        for x in sub_args_list:
            config.update({x: sub_args[x]})
        config.update({"input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class Dropout(tf.keras.layers.Layer):
    """Dropout Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 rate,
                 noise_shape=None,
                 seed=None,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as Activation."""
        super(Dropout, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._layer_keras = ks.layers.Dropout(rate=rate, noise_shape=noise_shape, seed=seed)

    def call(self, inputs, **kwargs):
        """Forward pass."""
        if isinstance(inputs, tf.RaggedTensor):
            if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
            value_tensor = inputs.values
            out_tensor = self._layer_keras(value_tensor, **kwargs)
            return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
        elif isinstance(inputs, list):
            if self.input_tensor_type not in ["disjoint", "values_partition"]:
                print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
            out = self._layer_keras(inputs[0], **kwargs)
            return [out] + inputs[1:]
        elif isinstance(inputs, tf.Tensor):
            if self.input_tensor_type not in ["Tensor", "tensor"]:
                print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
            return self._layer_keras(inputs, **kwargs)
        else:
            raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        """Update config."""
        config = super(Dropout, self).get_config()
        sub_args = self._layer_keras.get_config()
        sub_args_list = ["rate", "noise_shape", "seed"]
        for x in sub_args_list:
            config.update({x: sub_args[x]})
        config.update({"input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class LayerNormalization(tf.keras.layers.Layer):
    """LayerNormalization Wrapper Layer to support RaggedTensor input with ragged-rank=1."""
    def __init__(self,
                 axis=-1,
                 epsilon=0.001, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Initialize layer same as Activation."""
        super(LayerNormalization, self).__init__(**kwargs)
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._layer_keras = ks.layers.LayerNormalization(axis=axis, epsilon=epsilon, center=center, scale=scale,
                                                         beta_initializer=beta_initializer,
                                                         gamma_initializer=gamma_initializer,
                                                         beta_regularizer=beta_regularizer,
                                                         gamma_regularizer=gamma_regularizer,
                                                         beta_constraint=beta_constraint,
                                                         gamma_constraint=gamma_constraint)

    def call(self, inputs, **kwargs):
        """Forward pass."""
        if isinstance(inputs, tf.RaggedTensor):
            if self.input_tensor_type not in ["ragged", "RaggedTensor"]:
                print("Warning: Received RaggedTensor but tensor type specified as:", self.input_tensor_type)
            value_tensor = inputs.values
            out_tensor = self._layer_keras(value_tensor, **kwargs)
            return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
        elif isinstance(inputs, list):
            if self.input_tensor_type not in ["disjoint", "values_partition"]:
                print("Warning: Received input list but tensor type specified as:", self.input_tensor_type)
            out = self._layer_keras(inputs[0], **kwargs)
            return [out] + inputs[1:]
        elif isinstance(inputs, tf.Tensor):
            if self.input_tensor_type not in ["Tensor", "tensor"]:
                print("Warning: Received Tensor but tensor type specified as:", self.input_tensor_type)
            return self._layer_keras(inputs, **kwargs)
        else:
            raise NotImplementedError("Error: Unsupported tensor input type of ", inputs)

    def get_config(self):
        """Update config."""
        config = super(LayerNormalization, self).get_config()
        sub_args = self._layer_keras.get_config()
        sub_args_list = ["axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer",
                         "beta_regularizer", "gamma_regularizer", "beta_constraint", "gamma_constraint"]
        for x in sub_args_list:
            config.update({x: sub_args[x]})
        config.update({"input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config
