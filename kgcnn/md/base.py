import tensorflow as tf
from typing import Union, List

ks = tf.keras


class ModelPostprocessorsBase:
    pass


class KgcnnModelPredictor:

    def __init__(self,
                 model: ks.models.Model = None,
                 model_inputs: Union[list, dict] = None,
                 model_outputs: Union[list, dict] = None,
                 graph_preprocessors: List[dict] = None,
                 model_postprocessors=None):
        self.model = model
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs
        self.graph_preprocessors = graph_preprocessors
        self.model_postprocessors = model_postprocessors

    def _model_load(self, file_path: str) -> list:
        pass

    @staticmethod
    def _translate_properties(properties, translation) -> dict:

        if isinstance(translation, list):
            assert isinstance(properties, list), "With '%s' require list for '%s'." % (translation, properties)
            output = {key: properties[i] for i, key in enumerate(translation)}
        elif isinstance(translation, dict):
            assert isinstance(properties, dict), "With '%s' require dict for '%s'." % (translation, properties)
            output = {key: properties[value] for key, value in translation.items()}
        elif isinstance(translation, str):
            assert not isinstance(properties, (list, dict)), "Must be array for str '%s'." % properties
            output = {translation: properties}
        else:
            raise TypeError("'%s' output translation must be 'str', 'dict' or 'list'." % properties)
        return output

    def __call__(self, graph_list) -> dict:

        graph_list.map_list(self.graph_preprocessors)
        tensor_input = graph_list.tensor(self.model_inputs)

        try:
            tensor_output = self.model(tensor_input, training=False)
        except ValueError:
            tensor_output = self.model.predict(tensor_input)

        # Translate output
        tensor_dict = self._translate_properties(tensor_output, self.model_outputs)

        # Cast to numpy
        output_dict = {key: value.numpy() for key, value in tensor_dict.items()}

        return output_dict