import keras as ks
import keras.saving
from keras import ops
from typing import Union
from kgcnn.models.utils import get_model_class
from keras.saving import deserialize_keras_object, serialize_keras_object
from keras.backend import backend


# In keras 3.0.0 there is no `ops.gradient()` function yet.
# Backend specific gradient implementation in the following.
if backend() == "tensorflow":
    import tensorflow as tf
elif backend() == "torch":
    import torch
elif backend() == "jax":
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    raise NotImplementedError("Backend '%s' not supported for force model." % backend())


@ks.saving.register_keras_serializable(package='kgcnn', name='EnergyForceModel')
class EnergyForceModel(ks.models.Model):
    r"""Outer model to wrap a normal invariant GNN to predict forces from energy predictions via partial derivatives.
    The Force :math:`\vec{F_i}` on Atom :math:`i` is given by

    .. math::

        \vec{F_i} = - \vec{\nabla}_i E_{\text{total}}

    Note that the model simply returns the tensor type of the coordinate input for forces. No casting is done
    by this class. This means that the model returns a ragged, disjoint or padded tensor depending on the tensor
    type of the coordinates.
    """

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
                 use_batch_jacobian: bool = None,
                 name: str = None,
                 outputs: Union[dict, list] = None
                 ):
        """Initialize Force model with an energy model.

        Args:
            inputs (list): List of inputs as dictionary kwargs of keras input layers.
            model_energy (ks.models.Model, dict): Keras model os deserialization dictionary for a keras model.
            coordinate_input (int): Position of the coordinate input.
            output_as_dict (bool, tuple): Names for the output if a dictionary should be returned. Or simply a bool
                which will use the names "energy" and "force".
            ragged_validate (bool): Whether to validate ragged or jagged tensors.
            output_to_tensor: Deprecated.
            output_squeeze_states (bool): Whether to squeeze state/energy dimension for forces
                in case of a single energy value.
            nested_model_config (bool): Whether there is a config for the energy model.
            is_physical_force (bool): Whether to return the physical force, e.g. the negative gradient of the energy.
            use_batch_jacobian: Deprecated.
            name (str): Name of the model.
            outputs: List of outputs as dictionary kwargs similar to inputs.
        """
        super().__init__()
        if model_energy is None:
            raise ValueError("Require valid model in `model_energy` for force prediction.")
        # Input for model_energy.
        self._model_energy = model_energy
        self.name = name
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
            raise TypeError("Input `model_energy` must be dict or `ks.models.Model` . Can not deserialize model.")

        # Additional parameters of io and behavior of this class.
        self.ragged_validate = ragged_validate
        self.coordinate_input = coordinate_input
        self.output_squeeze_states = output_squeeze_states
        self.is_physical_force = is_physical_force
        self.nested_model_config = nested_model_config
        self._force_outputs = outputs

        self.output_as_dict = output_as_dict
        if isinstance(output_as_dict, bool):
            self.output_as_dict_use = output_as_dict
            self.output_as_dict_names = ("energy", "force")
        elif isinstance(output_as_dict, (list, tuple)):
            self.output_as_dict_use = True
            self.output_as_dict_names = (output_as_dict[0], output_as_dict[1])
        else:
            self.output_as_dict_use = False

        energy_output_config = outputs[self.output_as_dict_names[0]] if self.output_as_dict_use else outputs[0]
        self._expected_energy_states = energy_output_config["shape"][0]

        # We can try to infer the model inputs from energy model, if not given explicit.
        self._inputs_to_force_model = inputs
        if self._inputs_to_force_model is None:
            if self.nested_model_config and isinstance(model_energy, dict):
                self._inputs_to_force_model = model_energy["config"]["inputs"]

        if backend() == "tensorflow":
            self._call_grad_backend = self._call_grad_tf
        elif backend() == "torch":
            self._call_grad_backend = self._call_grad_torch
        elif backend() == "jax":
            self._call_grad_backend = self._call_grad_jax
        else:
            raise NotImplementedError("Backend '%s' not supported for force model." % backend())

    def build(self, input_shape):
        self.energy_model.build(input_shape)
        self.built = True

    def _call_grad_tf(self, inputs, training=False, **kwargs):

        x_in = inputs[self.coordinate_input]
        with tf.GradientTape(persistent=True) as tape:
            if isinstance(x_in, tf.RaggedTensor):
                x, splits = x_in.values, x_in.row_splits
            else:
                x, splits = x_in, None
            tape.watch(x)
            eng = self.energy_model.call(inputs, training=training, **kwargs)
            eng_sum = tf.reduce_sum(eng, axis=0, keepdims=False)
            e_grad = [eng_sum[i] for i in range(eng_sum.shape[-1])]
        e_grad = [tf.expand_dims(tape.gradient(e_i, x), axis=-1) for e_i in e_grad]
        e_grad = tf.concat(e_grad, axis=-1)

        if self.output_squeeze_states:
            e_grad = tf.squeeze(e_grad, axis=-1)

        if isinstance(x_in, tf.RaggedTensor):
            e_grad = tf.RaggedTensor.from_row_splits(e_grad, splits, validate=self.ragged_validate)

        return eng, e_grad

    def _call_grad_torch(self, inputs, training=False, **kwargs):

        x = inputs[self.coordinate_input]
        with torch.enable_grad():
            x.requires_grad = True
            eng = self.energy_model.call(inputs, training=training, **kwargs)
            eng_sum = eng.sum(dim=0)
            e_grad = torch.cat([
                torch.unsqueeze(torch.autograd.grad(eng_sum[i], x, create_graph=True, allow_unused=True)[0], dim=-1) for i in
                range(eng.shape[-1])], dim=-1)

        if self.output_squeeze_states:
            e_grad = torch.squeeze(e_grad, dim=-1)
        return eng, e_grad

    def _call_grad_jax(self, inputs, training=False, **kwargs):

        @partial(jax.jit, static_argnames=['pos'])
        def energy_reduce(*inputs, pos: int = 0):
            eng_temp = self.energy_model.call(inputs, training=training, **kwargs)
            eng_sum = jnp.sum(eng_temp, axis=0)[pos]
            return eng_sum

        grad_fn = jax.grad(energy_reduce, argnums=self.coordinate_input)
        all_grad = [grad_fn(*inputs, pos=i) for i in range(self._expected_energy_states)]
        eng = self.energy_model.call(inputs, training=training, **kwargs)
        e_grad = jnp.concatenate([jnp.expand_dims(x, axis=-1) for x in all_grad], axis=-1)

        if self.output_squeeze_states:
            e_grad = jnp.squeeze(e_grad, axis=-1)
        return eng, e_grad

    def call(self, inputs, training=False, **kwargs):

        eng, e_grad = self._call_grad_backend(inputs, training=training, **kwargs)

        if self.is_physical_force:
            e_grad = -e_grad

        if self.output_as_dict_use:
            return {self.output_as_dict_names[0]: eng, self.output_as_dict_names[1]: e_grad}
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
            # "output_to_tensor": self.output_to_tensor,
            "output_squeeze_states": self.output_squeeze_states,
            "nested_model_config": self.nested_model_config,
            # "use_batch_jacobian": self.use_batch_jacobian,
            "inputs": self._inputs_to_force_model,
            "outputs": self._force_outputs
        })
        return conf
