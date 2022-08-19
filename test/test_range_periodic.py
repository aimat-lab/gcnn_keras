import unittest
import numpy as np

from kgcnn.graph.geom import range_neighbour_lattice


class TestRangePeriodic(unittest.TestCase):

    # Handmade test case.
    artificial_lattice = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    artificial_atoms = np.array([[0.1, 0.0, 0.0], [0.5, 0.5, 0.5]])

    real_lattice = np.array([])
    real_atoms = np.array([])

    def test_nn_range(self):

        for x in [5, 10, 50]:
            indices, _, _ = range_neighbour_lattice(self.artificial_atoms,
                                                    self.artificial_lattice, max_distance=None,
                                                    max_neighbours=x)
            self.assertTrue(len(indices) == 2*x)


if __name__ == '__main__':
    TestRangePeriodic().test_nn_range()
    unittest.main()
