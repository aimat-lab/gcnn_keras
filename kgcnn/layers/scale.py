import keras as ks
from typing import Union
from kgcnn.layers.pooling import PoolingNodes
import numpy as np
from keras import ops


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
        self.extensive = False

        if self._scaling_shape is not None:
            self._add_weights_for_scaling()

    def _add_weights_for_scaling(self):
        self.scale_ = self.add_weight(
            shape=self._scaling_shape,
            initializer="ones", trainable=self._weights_trainable, dtype=self.dtype_scale
        )
        self.mean_ = self.add_weight(
            shape=self._scaling_shape,
            initializer="zeros", trainable=self._weights_trainable, dtype=self.dtype_scale
        )

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
        return ops.cast(inputs, dtype=self.dtype_scale)*self.scale_ + self.mean_

    def get_config(self):
        config = super(StandardLabelScaler, self).get_config()
        config.update({})
        return config

    def set_scale(self, scaler):
        self.set_weights([scaler.get_scaling(), scaler.get_mean_shift()])


class ExtensiveMolecularLabelScaler(ks.layers.Layer):  # noqa

    max_atomic_number = 95

    def __init__(self, scaling_shape: tuple = None, dtype_scale: str = "float64", trainable: bool = False,
                 name="ExtensiveMolecularLabelScaler", **kwargs):
        r"""Initialize layer instance of :obj:`StandardLabelScaler` .

        Args:
            scaling_shape (tuple): Shape
        """
        super(ExtensiveMolecularLabelScaler, self).__init__(**kwargs)
        self._scaling_shape = scaling_shape
        self.name = name
        self._weights_trainable = trainable
        self.dtype_scale = dtype_scale
        self.extensive = True
        self.layer_pool = PoolingNodes(pooling_method="scatter_sum")

        self._fit_atom_selection_mask = self.add_weight(
            shape=(self.max_atomic_number, ), trainable=False, dtype="bool", initializer="zeros")

        if self._scaling_shape is not None:
            self._add_weights_for_scaling()

    def _add_weights_for_scaling(self):
        self.scale_ = self.add_weight(
            shape=self._scaling_shape,
            initializer="ones", trainable=self._weights_trainable, dtype=self.dtype_scale
        )
        self.ridge_kernel_ = self.add_weight(
            shape=tuple([self.max_atomic_number] + list(self._scaling_shape[1:])),
            initializer="zeros", trainable=self._weights_trainable, dtype=self.dtype_scale
        )

    def build(self, input_shape):
        if self._scaling_shape is None:
            if input_shape is None:
                raise ValueError("Can not build scale and mean weights if `input_shape` and `scaling_shape` not known.")
            self._scaling_shape = tuple([1 if i is None else i for i in input_shape[0]])
            self._add_weights_for_scaling()

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        graph, nodes, batch_id = inputs
        energy_per_node = ops.take(self.ridge_kernel_, nodes, axis=0)
        extensive_energies = self.layer_pool([graph, energy_per_node, batch_id])
        return ops.cast(graph, dtype=self.dtype_scale)*self.scale_ + extensive_energies

    def get_config(self):
        config = super(ExtensiveMolecularLabelScaler, self).get_config()
        config.update({})
        return config

    def set_scale(self, scaler):
        ridge_kernel = np.transpose(np.array(scaler.ridge.coef_))
        pos = np.sort(np.array(scaler._fit_atom_selection))
        mask = np.array(scaler._fit_atom_selection_mask)
        shape = tuple([int(self.max_atomic_number)] + list(ridge_kernel.shape[1:]))
        layer_kernel = np.zeros(shape)
        layer_kernel[pos] = ridge_kernel
        layer_kernel[0] = 0.  # Make sure 0 is always 0.
        self.set_weights([mask, scaler.get_scaling(), layer_kernel])


class QMGraphLabelScaler(ks.layers.Layer):  # noqa

    max_atomic_number = 95

    def __init__(self, scaler_list: list = None, name="QMGraphLabelScaler", **kwargs):
        r"""Initialize layer instance of :obj:`StandardLabelScaler` .

        Args:
            scaler_list (list): List of scaler
        """
        super(QMGraphLabelScaler, self).__init__(**kwargs)
        self._scaler_list = scaler_list
        self.name = name
        self.extensive = True

    def build(self, input_shape):
        for scaler in self._scaler_list:
            if scaler.extensive:
                scaler.build(input_shape)
            else:
                scaler.build(input_shape[0])
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        return inputs

    def get_config(self):
        config = super(QMGraphLabelScaler, self).get_config()
        config.update({})
        return config

    def set_scale(self, scaler):
        for s in self._scaler_list:
            s.set_scale(scaler)


def get(scale_name: str):
    scaler_reference = {
        "StandardLabelScaler": StandardLabelScaler,
        "ExtensiveMolecularLabelScaler": ExtensiveMolecularLabelScaler
    }
    return scaler_reference[scale_name]