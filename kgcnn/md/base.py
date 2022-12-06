import tensorflow as tf
from typing import Union, List, Callable, Dict
from kgcnn.data.base import MemoryGraphList
from kgcnn.graph.base import GraphDict
from kgcnn.graph.serial import get_preprocessor
from kgcnn.utils.serial import deserialize, serialize

ks = tf.keras


class MolDynamicsModelPredictor:

    def __init__(self,
                 model: ks.models.Model = None,
                 model_inputs: Union[list, dict] = None,
                 model_outputs: Union[list, dict] = None,
                 graph_preprocessors: List[Callable] = None,
                 model_postprocessors: List[Callable] = None,
                 batch_size: int = 32):
        if graph_preprocessors is None:
            graph_preprocessors = []
        if model_postprocessors is None:
            model_postprocessors = []
        self.model = model
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs
        self.graph_preprocessors = [deserialize(gp) if isinstance(gp, dict) else gp for gp in graph_preprocessors]
        self.model_postprocessors = [deserialize(gp) if isinstance(gp, dict) else gp for gp in model_postprocessors]
        self.batch_size = batch_size

    def load(self, file_path: str) -> list:
        raise NotImplementedError("Not yet supported.")

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

    def __call__(self, graph_list: MemoryGraphList) -> MemoryGraphList:

        num_samples = len(graph_list)
        for gp in self.graph_preprocessors:
            for i in range(num_samples):
                graph_list[i].apply_preprocessor(gp)
        tensor_input = graph_list.tensor(self.model_inputs)

        try:
            tensor_output = self.model(tensor_input, training=False)
        except ValueError:
            tensor_output = self.model.predict(tensor_input, batch_size=self.batch_size)

        # Translate output. Mapping of model dict or list to dict for required calculator.
        tensor_dict = self._translate_properties(tensor_output, self.model_outputs)

        # Cast to numpy output and apply postprocessors.
        output_list = []
        for i in range(num_samples):
            temp_dict = {
                key: value[i].numpy() for key, value in tensor_dict.items()
            }
            temp_dict = GraphDict(temp_dict)
            for mp in self.model_postprocessors:
                post_temp = mp(graph=temp_dict, pre_graph=graph_list[i])
                temp_dict.update(post_temp)
            output_list.append(temp_dict)

        return MemoryGraphList(output_list)
