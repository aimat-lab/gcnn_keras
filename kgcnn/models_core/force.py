import keras_core as ks
import keras_core.saving
from typing import Union
from kgcnn.models_core.utils import get_model_class
from keras_core.saving.serialization_lib import deserialize_keras_object, serialize_keras_object
from keras_core.backend import backend


@ks.saving.register_keras_serializable(package='kgcnn', name='EnergyForceModel')
class EnergyForceModel(ks.models.Model):

    def __init__(self,
                 inputs: Union[dict, list] = None,
                 model_energy=None,
                 coordinate_input: Union[int, str] = 1,
                 output_as_dict: bool = True,
                 ragged_validate: bool = False,
                 output_to_tensor: bool = True,
                 output_squeeze_states: bool = False,
                 nested_model_config: bool = True,
                 is_physical_force: bool = True,
                 use_batch_jacobian: bool = True,
                 ):

        super().__init__()
        if model_energy is None:
            raise ValueError("Require valid model in `model_energy` for force prediction.")
        # Input for model_energy.
        self._model_energy = model_energy

        if isinstance(model_energy, ks.models.Model):
            # Ignoring module_name and class_name.
            self.energy_model = model_energy
        elif isinstance(model_energy, dict):
            if "module_name" not in model_energy:
                self.energy_model = deserialize_keras_object(model_energy)
            else:
                self.energy_model_class = get_model_class(model_energy["module_name"], model_energy["class_name"])
                self.energy_model = self.energy_model_class(**model_energy["config"])
        else:
            raise TypeError("Input `model_energy` must be dict or `ks.models.Model` .")

        # Additional parameters of io and behavior of this class.
        self.ragged_validate = ragged_validate
        self.coordinate_input = coordinate_input
        self.output_as_dict = output_as_dict
        self.output_to_tensor = output_to_tensor
        self.output_squeeze_states = output_squeeze_states
        self.is_physical_force = is_physical_force
        self.nested_model_config = nested_model_config
        self.use_batch_jacobian = use_batch_jacobian

        # We can try to infer the model inputs from energy model, if not given explicit.
        self._inputs_to_force_model = inputs

    def build(self, input_shape):
        self.energy_model.build(input_shape)
        self.built = True

    def call(self, inputs, training=False, **kwargs):

        x = inputs[self.coordinate_input]

        if backend() == "tensorflow":
            import tensorflow as tf
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                eng = self.energy_model(inputs, training=training, **kwargs)
                eng_sum = tf.reduce_sum(eng, axis=0, keepdims=False)
                e_grad = [eng_sum[i] for i in range(eng_sum.shape[-1])]
            e_grad = [tf.expand_dims(tape.gradient(e_i, x), axis=-1) for e_i in e_grad]
            e_grad = tf.concat(e_grad, axis=-1)
            if self.is_physical_force:
                e_grad = -e_grad
            if self.output_squeeze_states:
                e_grad = tf.squeeze(e_grad, axis=-1)
        elif backend() == "torch":
            import torch
            with torch.enable_grad():
                x.requires_grad = True
                eng = self.energy_model(inputs, training=training, **kwargs)
                eng_sum = eng.sum(dim=0)
                e_grad = torch.cat([
                    torch.unsqueeze(torch.autograd.grad(eng_sum[i], x, create_graph=True)[0], dim=-1) for i in
                    range(eng.shape[-1])], dim=-1)
            if self.is_physical_force:
                e_grad = -e_grad
            if self.output_squeeze_states:
                e_grad = torch.squeeze(e_grad, dim=-1)
        elif backend() == "jax":
            from jax import grad
            import jax.numpy as jnp
            e_grad = grad(self.energy_model, argnums=self.coordinate_input)(inputs)
            if self.is_physical_force:
                e_grad = -e_grad
            if self.output_squeeze_states:
                e_grad = jnp.squeeze(e_grad, axis=-1)
        else:
            raise NotImplementedError("Gradient not supported for backend '%s'." % backend())

        if self.output_as_dict:
            return {"energy": eng, "force": e_grad}
        else:
            return eng, e_grad

    def get_config(self):
        """Get config."""
        # Keras model does not provide config from base class.
        # conf = super(EnergyForceModel, self).get_config()
        conf = {}
        # Serialize class if _model_energy is not dict.
        if isinstance(self._model_energy, dict):
            model_energy = self._model_energy
        else:
            model_energy = serialize_keras_object(self._model_energy)
        conf.update({
            "model_energy": model_energy,
            "coordinate_input": self.coordinate_input,
            "output_as_dict": self.output_as_dict,
            "ragged_validate": self.ragged_validate,
            "output_to_tensor": self.output_to_tensor,
            "output_squeeze_states": self.output_squeeze_states,
            "nested_model_config": self.nested_model_config,
            "use_batch_jacobian": self.use_batch_jacobian,
            "inputs": self._inputs_to_force_model
        })
        return conf
