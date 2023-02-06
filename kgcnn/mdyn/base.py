import time
import tensorflow as tf
import numpy as np
from typing import Union, List, Callable, Dict
from kgcnn.data.base import MemoryGraphList
from kgcnn.graph.base import GraphDict
from kgcnn.utils.serial import deserialize, serialize

ks = tf.keras


class MolDynamicsModelPredictor:
    r"""Model predictor class that adds pre- and postprocessors to the keras model to be able to add transformation
    steps to convert for example input and output representations to fit MD programs like :obj:`ase` .
    The :obj:`MolDynamicsModelPredictor` receives a :obj:`MemoryGraphList` in call and returns a
    :obj:`MemoryGraphList` .


    """

    def __init__(self,
                 model: ks.models.Model = None,
                 model_inputs: Union[list, dict] = None,
                 model_outputs: Union[list, dict] = None,
                 graph_preprocessors: List[Callable] = None,
                 model_postprocessors: List[Callable] = None,
                 use_predict: bool = False,
                 batch_size: int = 32):
        r"""Initialize :obj:`MolDynamicsModelPredictor` class.

        Args:
            model (tf.keras.Model): Single trained keras model.
            model_inputs (list, dict): List or single dictionary for model inputs.
            model_outputs (list, dict): List of model output names or dictionary of output mappings from keras model
                output to the names in the return :obj:`GraphDict` .
            graph_preprocessors (list): List of graph preprocessors, see :obj:`kgcnn.graph.preprocessor` .
            model_postprocessors (list): List of graph postprocessors, see :obj:`kgcnn.graph.postprocessor` .
            use_predict (bool): Whether to use :obj:`model.predict()` or call method :obj:`model()` .
            batch_size (int): Optional batch size for prediction.
        """
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
        self.use_predict = use_predict

    def load(self, file_path: str):
        raise NotImplementedError("Not yet supported.")

    def save(self, file_path: str):
        raise NotImplementedError("Not yet supported.")

    @staticmethod
    def _translate_properties(properties, translation) -> dict:
        """Translate general model output.

        Args:
            properties (list, dict): List of properties or dictionary of properties.
            translation (str, list, dict): List of names or dictionary of name mappings like '{new_name: old_name}'.

        Returns:
            dict: Return dictionary with keys from translation.
        """

        if isinstance(translation, list):
            assert isinstance(properties, list), "With '%s' require list for '%s'." % (translation, properties)
            output = {key: properties[i] for i, key in enumerate(translation)}
        elif isinstance(translation, dict):
            assert isinstance(properties, dict), "With '%s' require dict for '%s'." % (translation, properties)
            output = {key: properties[value] for key, value in translation.items()}
        elif isinstance(translation, str):
            assert not isinstance(properties, (list, dict)), "Must be array-like for str '%s'." % properties
            output = {translation: properties}
        else:
            raise TypeError("'%s' output translation must be 'str', 'dict' or 'list'." % properties)
        return output

    def __call__(self, graph_list: MemoryGraphList) -> MemoryGraphList:
        """Prediction of the model wrapper.

        Args:
            graph_list (MemoryGraphList): List of graphs to predict e.g. energies and forces.

        Returns:
            MemoryGraphList: List of general return graph dictionaries from model output.
        """

        num_samples = len(graph_list)
        for gp in self.graph_preprocessors:
            for i in range(num_samples):
                graph_list[i].apply_preprocessor(gp)
        tensor_input = graph_list.tensor(self.model_inputs)

        if not self.use_predict:
            tensor_output = self.model(tensor_input, training=False)
        else:
            tensor_output = self.model.predict(tensor_input, batch_size=self.batch_size)

        # Translate output. Mapping of model dict or list to dict for required calculator.
        tensor_dict = self._translate_properties(tensor_output, self.model_outputs)

        # Cast to numpy output and apply postprocessors.
        output_list = []
        for i in range(num_samples):
            temp_dict = {
                key: np.array(value[i]) for key, value in tensor_dict.items()
            }
            temp_dict = GraphDict(temp_dict)
            for mp in self.model_postprocessors:
                post_temp = mp(graph=temp_dict, pre_graph=graph_list[i])
                temp_dict.update(post_temp)
            output_list.append(temp_dict)

        return MemoryGraphList(output_list)

    def _test_timing(self, graph_list: MemoryGraphList, repetitions: int = 100) -> float:
        """Evaluate timing for prediction.

        Args:
            graph_list (MemoryGraphList): List of graphs to predict e.g. energies and forces.

        Returns:
            float: Time for one call.
        """
        assert repetitions >= 1, "Repetitions must be number of calls."
        start = time.process_time()
        for _ in range(repetitions):
            output = self.__call__(graph_list)
        stop = time.process_time()
        return float(stop-start)/repetitions
