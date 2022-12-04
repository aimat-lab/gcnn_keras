import unittest

import numpy as np

from kgcnn.xai.utils import flatten_importances_list


class TestFunctions(unittest.TestCase):

    def test_flatten_importances_list_basically_works(self):
        importances_list = [
            np.array([
                [0, 1],
                [1, 0]
            ]),
            np.array([
                [0, 0, 0],
                [1, 0, 0]
            ])
        ]
        expected = [0, 1, 1, 0, 0, 0, 0, 1, 0, 0]

        result = flatten_importances_list(importances_list)
        self.assertListEqual(expected, result)
