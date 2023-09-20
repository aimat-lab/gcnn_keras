import keras_core as ks
import keras_core.saving
from typing import Union
from kgcnn.models_core.utils import get_model_class
from keras_core.saving.serialization_lib import deserialize_keras_object


@ks.saving.register_keras_serializable(package='kgcnn', name='EnergyForceModel')
def make_energy_force_model(
        inputs: dict = None,
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

    model_inputs = [ks.layers.Input(**x) for x in inputs]

    if model_energy is None:
        raise ValueError("Require valid model in `model_energy` for force prediction.")

    if isinstance(model_energy, dict):
        if "module_name" not in model_energy:
            model_energy = deserialize_keras_object(model_energy)
        else:
            model_energy_class = get_model_class(model_energy["module_name"], model_energy["class_name"])
            model_energy = model_energy_class(**model_energy["config"])
    else:
        raise TypeError("Input `model_energy` must be dict or `ks.models.Model` .")

    out = model_energy(model_inputs)


    model = ks.models.Model(inputs=model_inputs, outputs=out, name="%s_ForceGradient" % model_energy.name)

    return model
