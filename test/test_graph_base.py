import unittest

import numpy as np

from kgcnn.graph.base import GraphDict


class TestGraphDict(unittest.TestCase):

    def test_creating_empty_one_basically_works(self):
        """
        If it is possible to create a GraphDict instance with empty constructor
        """
        g = GraphDict()
        self.assertIsInstance(g, GraphDict)
        self.assertEqual(0, len(g))

    def test_creating_graph_dict_from_existing_dict_works(self):
        """
        If it is basically possible to create a GraphDict object by passing an already existing dict to
        the constructor
        """
        data = {
            'graph_labels': np.array([0.1]),
            'node_indices': np.array([0, 1, 2]),
            'node_attributes': np.array([[1], [1], [1]]),
            'edge_indices': np.array([
                [0, 1], [1, 0],
                [1, 2], [2, 1],
                [2, 0], [0, 2]
            ]),
            'edge_attributes': np.array([[1], [1], [1], [1], [1], [1]])
        }
        g = GraphDict(data)
        self.assertIsInstance(g, GraphDict)
        self.assertIsInstance(g['node_attributes'], np.ndarray)
        self.assertEqual((3, ), g['node_indices'].shape)
        self.assertEqual((3, 1), g['node_attributes'].shape)
