import tensorflow as tf
from typing import Union
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.model.utils import get_model_class

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='EnergyForceModel')
class EnergyForceModel(ks.models.Model):
    r"""Force model that generates forces from any energy predicting model by taking the derivative with respect to
    the input coordinates.

    For now the model has to cast to dense tensor for using :obj:`batch_jacobian` , however, this will likely support
    ragged tensors in the future.

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.model.force import EnergyForceModel
        model = EnergyForceModel(
            module_name="kgcnn.literature.Schnet",
            class_name="make_model",
            config={
                "name": "SchnetEnergy",
                "inputs": [
                    {"shape": [None], "name": "z", "dtype": "float32", "ragged": True},
                    {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 128}
                },
                "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                             "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                "interaction_args": {
                    "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
                },
                "node_pooling_args": {"pooling_method": "sum"},
                "depth": 6,
                "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                "output_embedding": "graph",
                "use_output_mlp": False,
                "output_mlp": None,
            },
            coordinate_input=1,
            output_as_dict=True,
            output_to_tensor=False,
            output_squeeze_states=True,
            is_physical_force=True
        )

    """

    def __init__(self, module_name: str, class_name: str, config: dict, coordinate_input: Union[int, str] = 1,
                 output_as_dict: bool = True, ragged_validate: bool = False, output_to_tensor: bool = True,
                 output_squeeze_states: bool = False, nested_model_config: bool = True, is_physical_force: bool = True,
                 **kwargs):
        r"""Initialize :obj:`EnergyForceModel` with sub-model for energy prediction.

        This wrapper model was designed for models in `kgcnn.literature` that predict energy from geometric
        information.

        .. note::

            The energy model is inferred by `module_name` , `class_name` , `config` within :obj:`kgcnn` , but you can
            also pass an external model directly to `class_name` and set `module_name` and `config` to `None` and
            set `nested_model_config` to `False` or pass its config dictionary manually.

        Args:
            module_name (str): Module where to find energy model. Can be `None` for custom keras model.
            class_name (str): Class name of energy model. Can also be full keras model instead of class name.
                In this case module_name is ignored and should be set to `None` .
            config (dict): Config of energy model.
            coordinate_input (str, int): Index or key where to find coordinate tensor in model input.
            output_as_dict (bool): Whether to return energy and force as list or as dict. Default is True.
            ragged_validate (bool): Whether to validate ragged tensor creation. Default is False.
            output_to_tensor (bool): Whether to cast the output to tensor or keep ragged output. Default is True
            output_squeeze_states (bool): Whether to squeeze states, which can be done for one energy value to remove
                an axis of one.
            nested_model_config (bool): Whether `config` has model config of the energy model. Default is True.
            is_physical_force (bool): Whether gradient of force, which is the negative gradient, is to be returned.
        """
        super(EnergyForceModel, self).__init__(self, **kwargs)
        self.model_config = config
        self.ragged_validate = ragged_validate
        self.module_name = module_name
        self.class_name = class_name
        if isinstance(class_name, ks.models.Model):
            # Ignoring module_name and class_name.
            self.energy_model = class_name
        elif isinstance(class_name, dict):
            self.energy_model = ks.utils.deserialize_keras_object(class_name)
        else:
            self.energy_model_class = get_model_class(module_name, class_name)
            self.energy_model = self.energy_model_class(**self.model_config)
        self.coordinate_input = coordinate_input
        self.cast_coordinates = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask")
        self.output_as_dict = output_as_dict
        self.output_to_tensor = output_to_tensor
        self.output_squeeze_states = output_squeeze_states
        self.is_physical_force = is_physical_force
        self.nested_model_config = nested_model_config

    def call(self, inputs, training=False, **kwargs):
        """Forward pass that wraps energy model in gradient tape.

        Args:
            inputs (list, dict): Must be list of (tensor) input for energy model.
                Index or key to find coordinates must be provided.
            training (bool): Whether model is in training, passed to energy model. Default is False.

        Returns:
            dict, list: Model output plus force or derivative.
        """
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
        """Get config."""
        # Keras model does not provide config from base class.
        # conf = super(EnergyForceModel, self).get_config()
        conf = {}
        # Serialize class if class_name is not string.
        if isinstance(self.class_name, str):
            class_name = self.class_name
        else:
            class_name = ks.utils.serialize_keras_object(self.class_name)
        conf.update({
            "module_name": self.module_name,
            "class_name": class_name,
            "config": self.model_config,
            "coordinate_input": self.coordinate_input,
            "output_as_dict": self.output_as_dict,
            "ragged_validate": self.ragged_validate,
            "output_to_tensor": self.output_to_tensor,
            "output_squeeze_states": self.output_squeeze_states,
            "nested_model_config": self.nested_model_config
        })
        return conf
