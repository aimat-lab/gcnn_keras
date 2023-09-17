import keras_core as ks
from typing import Union
from keras_core import ops


class StandardLabelScaler(ks.layers.Layer):  # noqa

    def __init__(self, scaling_shape: tuple = None, dtype_scale: str = "float64", trainable: bool = False,
                 name="StandardLabelScaler", **kwargs):
        r"""Initialize layer instance of :obj:`StandardLabelScaler` .

        Args:
            scaling_shape (tuple): Shape
        """
        super(StandardLabelScaler, self).__init__(**kwargs)
        self._scaling_shape = scaling_shape
        self.name = name
        self._weights_trainable = trainable
        self.dtype_scale = dtype_scale
        if self._scaling_shape is not None:
            self._add_weights_for_scaling()

    def _add_weights_for_scaling(self):
        self.scale_ = self.add_weight(shape=self._scaling_shape, initializer="ones", trainable=self._weights_trainable,
                                      dtype=self.dtype_scale)
        self.mean_ = self.add_weight(shape=self._scaling_shape, initializer="zeros", trainable=self._weights_trainable,
                                      dtype=self.dtype_scale)

    def build(self, input_shape):
        if self._scaling_shape is None:
            if input_shape is None:
                raise ValueError("Can not build scale and mean weights if `input_shape` not known.")
            self._scaling_shape = tuple([1 if i is None else i for i in input_shape])
            self._add_weights_for_scaling()
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        r"""Forward pass of :obj:`StandardLabelScaler` .

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Statically re-scaled input tensor.
        """
        return ops.cast(inputs, dtype=self.dtype_scale)*self.scale_ + self.mean_

    def get_config(self):
        """Update config for `NodePosition`."""
        config = super(StandardLabelScaler, self).get_config()
        config.update({})
        return config

    def set_scale(self, scaler):
        self.set_weights([scaler.get_scaling(), scaler.get_mean_shift()])


def get(scale_layer: Union[dict, str]):
    return StandardLabelScaler