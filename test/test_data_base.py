import unittest
import random
import typing as t

import numpy as np

from kgcnn.graph.base import GraphDict
from kgcnn.data.base import MemoryGraphDataset


class TestMemoryGraphDataset(unittest.TestCase):

    # -- UTILITY METHODS --

    def create_dataset(self,
                       num_elements: int = 10,
                       train_ratio: float = 0.8,
                       num_splits: int = 1,
                       ) -> MemoryGraphDataset:
        memory_dataset = MemoryGraphDataset()
        memory_dataset.empty(num_elements)

        indices = list(range(num_elements))
        train_indices_list = [random.sample(indices, k=int(num_elements * train_ratio))
                              for _ in range(num_splits)]
        test_indices_list = [[index for index in indices if index not in train_indices]
                             for train_indices in train_indices_list]

        for index in range(num_elements):
            graph_dict = GraphDict({
                'node_indices': np.array([0, 1, 2]),
                'node_attributes': np.array([
                    [random.random()],
                    [random.random()],
                    [random.random()],
                ]),
                'edge_indices': np.array([[0, 1], [1, 2], [2, 3]]),
                'edge_attributes': np.array([
                    [random.random()],
                    [random.random()],
                    [random.random()]
                ]),
                'graph_labels': [random.random(), random.random()],
                'train': [(i + 1)
                          for i, train_indices in enumerate(train_indices_list)
                          if index in train_indices],
                'test': [(i + 1)
                         for i, test_indices in enumerate(test_indices_list)
                         if index in test_indices],
            })
            memory_dataset[index] = graph_dict

        return memory_dataset

    # -- UNITTESTS --

    def test_creating_from_list_of_graph_dicts_basically_works(self):
        """
        If it is possible to create a MemoryGraphDataset from a list of GraphDicts
        """
        num_elements = 10

        memory_dataset = MemoryGraphDataset()
        memory_dataset.empty(num_elements)
        for i in range(num_elements):
            graph_dict = GraphDict({
                'node_indices': np.array([0, 1, 2]),
                'node_attributes': np.array([[random.random(), random.random(), random.random()]])
            })
            memory_dataset[i] = graph_dict

        self.assertIsInstance(memory_dataset, MemoryGraphDataset)
        self.assertEqual(num_elements, len(memory_dataset))

        node_indices_list = memory_dataset.obtain_property('node_indices')
        self.assertIsInstance(node_indices_list, list)
        self.assertEqual(num_elements, len(node_indices_list))

    def test_create_dataset_works(self):
        """
        If the utility method "create_dataset" of this class works as intended
        """
        num_elements = 10
        memory_dataset: MemoryGraphDataset = self.create_dataset(num_elements)
        self.assertIsInstance(memory_dataset, MemoryGraphDataset)
        self.assertEqual(num_elements, len(memory_dataset))

    def test_get_train_test_indices_works(self):
        num_elements = 10
        train_ratio = 0.8
        memory_dataset: MemoryGraphDataset = self.create_dataset(
            num_elements=num_elements,
            train_ratio=train_ratio,
            num_splits=1,
        )

        train_test_indices = memory_dataset.get_train_test_indices(
            train='train',
            test='test',
            valid=None
        )
        train_indices, test_indices = train_test_indices[0]
        self.assertIsInstance(train_indices, np.ndarray)
        self.assertEqual(round(train_ratio * num_elements), len(train_indices))

        self.assertIsInstance(test_indices, np.ndarray)
        self.assertEqual(round((1 - train_ratio) * num_elements), len(test_indices))

    def test_get_train_test_indices_with_multiple_splits_works(self):
        num_elements = 10
        train_ratio = 0.8
        num_splits = 3
        memory_dataset: MemoryGraphDataset = self.create_dataset(
            num_elements=num_elements,
            train_ratio=train_ratio,
            num_splits=num_splits
        )

        train_test_indices = memory_dataset.get_train_test_indices(
            train='train',
            test='test',
            valid=None,
            split_index=list(range(1, num_splits + 1))
        )
        self.assertEqual(num_splits, len(train_test_indices))
        for train_indices, test_indices in train_test_indices:

            self.assertIsInstance(train_indices, np.ndarray)
            self.assertEqual(round(train_ratio * num_elements), len(train_indices))

            self.assertIsInstance(test_indices, np.ndarray)
            self.assertEqual(round((1 - train_ratio) * num_elements), len(test_indices))


