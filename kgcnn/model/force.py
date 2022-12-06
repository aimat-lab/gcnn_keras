import tensorflow as tf
from typing import Union
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.model.utils import get_model_class

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='EnergyForceModel')
class EnergyForceModel(ks.models.Model):
    def __init__(self, module_name, class_name, config, coordinate_input: Union[int, str] = 1,
                 output_as_dict: bool = True, ragged_validate: bool = False, output_to_tensor: bool = True,
                 output_squeeze_states: bool = False, nested_model_config: bool = True, is_physical_force: bool = True,
                 **kwargs):
        super(EnergyForceModel, self).__init__(self, **kwargs)
        self.model_config = config
        self.ragged_validate = ragged_validate
        self.energy_model_class = get_model_class(module_name, class_name)
        self.energy_model = self.energy_model_class(**self.model_config)
        self.coordinate_input = coordinate_input
        self.cast_coordinates = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask")
        self.output_as_dict = output_as_dict
        self.output_to_tensor = output_to_tensor
        self.output_squeeze_states = output_squeeze_states
        self.is_physical_force = is_physical_force

    def call(self, inputs, training=False, **kwargs):
        x = inputs[self.coordinate_input]
        inputs_energy = [i for i in inputs]
        # x is ragged tensor of shape (batch, [N], 3) with cartesian coordinates.
        # `batch_jacobian` does not yet support ragged tensor input.
        # Cast to masked tensor for coordinates only.
        x_pad, x_mask = self.cast_coordinates(x, training=training, **kwargs)  # (batch, N, 3), (batch, N, 3)
        with tf.GradientTape() as tape:
            tape.watch(x_pad)
            # Temporary solution for casting.
            # Cast back to ragged tensor for model input.
            x_pad_to_ragged = self._cast_coordinates_pad_to_ragged(x_pad, x_mask, self.ragged_validate)
            inputs_energy[self.coordinate_input] = x_pad_to_ragged
            # Predict energy.
            # Energy must be tensor of shape (batch, states)
            eng = self.energy_model(inputs_energy, training=training, **kwargs)
        e_grad = tape.batch_jacobian(eng, x_pad)
        e_grad = tf.transpose(e_grad, perm=[0, 2, 3, 1])

        if self.is_physical_force:
            e_grad = -e_grad

        if self.output_squeeze_states:
            e_grad = tf.squeeze(e_grad, axis=-1)
        if not self.output_to_tensor:
            e_grad = self._cast_coordinates_pad_to_ragged(e_grad, x_mask, self.ragged_validate)
        if self.output_as_dict:
            return {"energy": eng, "force": e_grad}
        else:
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
