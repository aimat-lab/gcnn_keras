import tensorflow as tf
from typing import Union
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.utils.models import get_model_class
ks = tf.keras


class EnergyForceModel(ks.models.Model):
    def __init__(self, module_name, class_name, config, coordinate_input: Union[int, str] = 1,
                 output_as_dict: bool = False, ragged_validate: bool = False, **kwargs):
        super(EnergyForceModel, self).__init__(self, **kwargs)
        self.model_config = config
        self.ragged_validate = ragged_validate
        self.energy_model_class = get_model_class(module_name, class_name)
        self.energy_model = self.energy_model_class(**self.model_config)
        self.coordinate_input = coordinate_input
        self.cast_coordinates = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask")

    def call(self, inputs, training=False, **kwargs):
        x = inputs[self.coordinate_input]
        inputs_energy = [i for i in inputs]
        # x is ragged tensor of shape (batch, [N], 3) with cartesian coordinates.
        # `batch_jacobian` does not yet support ragged tensor input.
        # Cast to masked tensor for coordinates only.
        x_pad, x_mask = self.cast_coordinates(x, training=training)  # (batch, N, 3), (batch, N, 3)
        with tf.GradientTape() as tape:
            tape.watch(x_pad)
            # Temporary solution for casting.
            # Cast back to ragged tensor for model input.
            x_pad_to_ragged = self._cast_coordinates_pad_to_ragged(x_pad, x_mask, self.ragged_validate)
            inputs_energy[self.coordinate_input] = x_pad_to_ragged
            # Predict energy.
            # Energy must be tensor of shape (batch, States)
            eng = self.energy_model(inputs_energy, training=training)
        e_grad = tape.batch_jacobian(eng, x_pad)

        return eng, e_grad

    # Temporary solution.
    @staticmethod
    @tf.function
    def _cast_coordinates_pad_to_ragged(x_pad, x_mask, validate):
        # Assume that float mask is the same for all coordinates.
        x_mask_number = tf.cast(x_mask[:, :, 0], dtype="bool")  # (batch, N)
        x_values = x_pad[x_mask_number]
        x_row_length = tf.reduce_sum(tf.cast(x_mask_number, dtype="int64"), axis=-1)
        return tf.RaggedTensor.from_row_lengths(x_values, x_row_length, validate=validate)

    def get_config(self):
        # conf = super(EnergyForceModel, self).get_config()
        conf = {}
        conf.update({})
        return conf
