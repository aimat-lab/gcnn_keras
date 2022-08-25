import numpy as np
import tensorflow as tf
from typing import Union, List
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.data.base import MemoryGraphDataset
ks = tf.keras


class GraphBatchLoader(ks.utils.Sequence):
    r"""Example (minimal) implementation of a graph batch loader based on :obj:`ks.utils.Sequence`."""

    def __init__(self,
                 data: Union[List[dict], MemoryGraphDataset],
                 inputs: Union[dict, List[dict]],
                 outputs: Union[dict, List[dict]],
                 batch_size: int = 32,
                 shuffle: bool = False):
        """Initialization with data and input information.

        Args:
            data (list, MemoryGraphDataset): Any iterable data that implements indexing operator for graph instance.
                Each graph instance must implement indexing operator for named property.
            inputs (dict, list):  List of dictionaries that specify graph properties in list via 'name' key.
                The dict-items match the tensor input for :obj:`tf.keras.layers.Input` layers.
                Required dict-keys should be 'name' and 'ragged'.
                Optionally shape information can be included via 'shape'.
                E.g.: `[{'name': 'edge_indices', 'ragged': True}, {...}, ...]`.
            outputs (dict, list): List of dictionaries that specify graph properties in list via 'name' key.
                Required dict-keys should be 'name' and 'ragged'.
                Optionally shape information can be included via 'shape'.
                E.g.: `[{'name': 'graph_labels', 'ragged': False}, {...}, ...]`.
            batch_size (int): Batch size. Default is 32.
            shuffle (bool): Whether to shuffle data. Default is False.
        """
        self.data = data
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))

        self._shuffle_indices()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x_model, y_model = self._data_generation(batch_indices)

        # return batch_indexes
        return x_model, y_model

    def _shuffle_indices(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        """Updates after each epoch"""
        self._shuffle_indices()

    @staticmethod
    def _to_tensor(item: Union[np.ndarray, list], is_ragged: bool):
        if is_ragged:
            return ragged_tensor_from_nested_numpy(item)
        else:
            return tf.constant(np.array(item))

    def _data_generation(self, batch_indices: Union[np.ndarray, list]):
        """Generates data containing batch_size samples"""
        graphs = [self.data[int(i)] for i in batch_indices]
        # Inputs
        inputs = self.inputs if not isinstance(self.inputs, dict) else [self.inputs]
        x_inputs = []
        for i in inputs:
            data_list = [g[i["name"]] for g in graphs]
            is_ragged = i["ragged"] if "ragged" in i else False
            x_inputs.append(self._to_tensor(data_list, is_ragged))
        y_outputs = []
        # Outputs
        outputs = self.outputs if not isinstance(self.outputs, dict) else [self.outputs]
        for i in outputs:
            data_list = [g[i["name"]] for g in graphs]
            is_ragged = i["ragged"] if "ragged" in i else False
            y_outputs.append(self._to_tensor(data_list, is_ragged))
        # Check return type.
        x_inputs = x_inputs if not isinstance(self.inputs, dict) else x_inputs[0]
        y_outputs = y_outputs if not isinstance(self.outputs, dict) else y_outputs[0]
        return x_inputs, y_outputs
