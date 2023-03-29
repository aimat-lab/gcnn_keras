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
                 graph_postprocessors: List[Callable] = None,
                 store_last_input: bool = False,
                 store_last_output: bool = False,
                 copy_graphs_in_store: bool = False,
                 use_predict: bool = False,
                 predict_verbose: Union[str, int] = 0,
                 batch_size: int = 32,
                 update_from_last_input: list = None,
                 update_from_last_input_skip: int = None,
                 ):
        r"""Initialize :obj:`MolDynamicsModelPredictor` class.

        Args:
            model (tf.keras.Model): Single trained keras model.
            model_inputs (list, dict): List or single dictionary for model inputs.
            model_outputs (list, dict): List of model output names or dictionary of output mappings from keras model
                output to the names in the return :obj:`GraphDict` .
            graph_preprocessors (list): List of graph preprocessors, see :obj:`kgcnn.graph.preprocessor` .
            graph_postprocessors (list): List of graph postprocessors, see :obj:`kgcnn.graph.postprocessor` .
            use_predict (bool): Whether to use :obj:`model.predict()` or call method :obj:`model()` .
            batch_size (int): Optional batch size for prediction.
            store_last_input (bool): Whether to store last input graph list for model input. Default is False.
            store_last_output (bool): Whether to store last output graph list from model output. Default is False.
            copy_graphs_in_store (bool): Whether to make a copy of the graph lists or keep reference. Default is False.
            update_from_last_input (list): List of graph properties to copy from last input into current input.
                This is placed before graph preprocessors. Default is None.
            update_from_last_input_skip (int): If set to a value, this will skip the update from last input at
                given number of calls. Uses counter. Default is None.
        """
        if graph_preprocessors is None:
            graph_preprocessors = []
        if graph_postprocessors is None:
            graph_postprocessors = []
        self.model = model
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs
        self.graph_preprocessors = [deserialize(gp) if isinstance(gp, dict) else gp for gp in graph_preprocessors]
        self.graph_postprocessors = [deserialize(gp) if isinstance(gp, dict) else gp for gp in graph_postprocessors]
        self.batch_size = batch_size
        self.use_predict = use_predict
        self.store_last_input = store_last_input
        self.store_last_output = store_last_output
        self.copy_graphs_in_store = copy_graphs_in_store
        self.update_from_last_input = update_from_last_input
        self.update_from_last_input_skip = update_from_last_input_skip
        self.predict_verbose = predict_verbose

        self._last_input = None
        self._last_output = None
        self._counter = 0

    def load(self, file_path: str):
        raise NotImplementedError("Not yet supported.")

    def save(self, file_path: str):
        raise NotImplementedError("Not yet supported.")

    @tf.function
    def _call_model_(self, tensor_input):
        return self.model(tensor_input, training=False)

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

        skip = self._counter % self.update_from_last_input_skip == 0 if self.update_from_last_input_skip else False
        if self.update_from_last_input is not None and self._last_input is not None and not skip:
            for i in range(num_samples):
                for prop in self.update_from_last_input:
                    graph_list[i].set(prop, self._last_input[i].get(prop))

        for gp in self.graph_preprocessors:
            for i in range(num_samples):
                graph_list[i].apply_preprocessor(gp)

        if self.store_last_input:
            if self.copy_graphs_in_store:
                self._last_input = graph_list.copy()
            else:
                self._last_input = graph_list

        tensor_input = graph_list.tensor(self.model_inputs)

        if not self.use_predict:
            tensor_output = self._call_model_(tensor_input)
        else:
            tensor_output = self.model.predict(tensor_input, batch_size=self.batch_size, verbose=self.predict_verbose)

        # Translate output. Mapping of model dict or list to dict for required calculator.
        tensor_dict = self._translate_properties(tensor_output, self.model_outputs)

        # Cast to numpy output and apply postprocessors.
        output_list = []
        for i in range(num_samples):
            temp_dict = {
                key: np.array(value[i]) for key, value in tensor_dict.items()
            }
            temp_dict = GraphDict(temp_dict)
            for mp in self.graph_postprocessors:
                post_temp = mp(graph=temp_dict, pre_graph=graph_list[i])
                temp_dict.update(post_temp)
            output_list.append(temp_dict)

        if self.store_last_output:
            if self.copy_graphs_in_store:
                self._last_output = output_list.copy()
            else:
                self._last_output = output_list

        # Increase counter.
        self._counter += 1

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
            self.__call__(graph_list)
        stop = time.process_time()
        return float(stop-start)/repetitions
