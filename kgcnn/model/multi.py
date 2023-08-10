import tensorflow as tf
import itertools
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.mlp import MLP

ks = tf.keras


def merge_models(model_list: list,
                 merge_type: str = "concat",
                 output_mlp: dict = None):
    r"""Merge a list of models by combining their output.

    Args:
        model_list (list): List of graph models.
        merge_type (str): How to merge the output.
        output_mlp (dict): Kwargs of the final MLP after the models' output.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    if output_mlp:
        if isinstance(output_mlp, dict):
            output_mlp = MLP(**output_mlp)

    combined_inputs = []
    for m in model_list:
        new_inputs_per_model = []
        for i, input_layer in enumerate(m.inputs):
            new_input_layer = ks.Input(type_spec=input_layer.type_spec, name=input_layer.name)
            new_inputs_per_model.append(new_input_layer)
        combined_inputs.append(new_inputs_per_model)

    new_outputs = []
    for x, m in zip(combined_inputs, model_list):
        new_outputs.append(m(x))

    if merge_type in ["concat", "concatenate"]:
        output = LazyConcatenate(axis=-1)(new_outputs)
    else:
        raise NotImplementedError("Unknown merge type '%s' for models" % merge_type)

    if output_mlp:
        output = output_mlp(output)

    flatten_inputs = list(itertools.chain(*combined_inputs))
    return ks.models.Model(inputs=flatten_inputs, outputs=output)
