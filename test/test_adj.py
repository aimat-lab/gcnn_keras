import numpy as np
import unittest

from kgcnn.utils.adj import add_edges_reverse_indices, add_self_loops_to_edge_indices
from kgcnn.utils.adj import get_angle_indices, get_angle, get_angle_between_edges


class ReverseEdges(unittest.TestCase):

    def test_add_indices_only_no_order(self):
        indices = np.array([[0, 0], [1, 2], [2, 3], [0, 1]])
        result = add_edges_reverse_indices(indices)
        expected_result = np.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]])
        # print(result)
        self.assertTrue(np.max(np.abs(result - expected_result)) < 1e-6)

    def test_add_edges_multiple(self):
        indices = np.array([[0, 0], [1, 2], [0, 1]])
        edge1 = np.array([1, 2, 3])
        edge2 = np.array([[11], [22], [33]])
        result, result1, result2 = add_edges_reverse_indices(indices, edge1, edge2)
        expected_result1 = np.array([1, 3, 3, 2, 2])
        self.assertTrue(np.max(np.abs(result1 - expected_result1)) < 1e-6)


class SelfLoops(unittest.TestCase):

    def test_indices_only_no_order(self):
        indices = np.array([[0, 0], [1, 2], [2, 3], [0, 1]])
        result = add_self_loops_to_edge_indices(indices)
        expected_result = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3]])
        # print(result)
        self.assertTrue(np.max(np.abs(result - expected_result)) < 1e-6)

    def test_add_edges_multiple(self):
        indices = np.array([[0, 0], [1, 2], [0, 1]])
        edge1 = np.array([1, 2, 3])
        edge2 = np.array([[11], [22], [33]])
        result, result1, result2 = add_self_loops_to_edge_indices(indices, edge1, edge2)
        # print(result, result1, result2)
        expected_result1 = np.array([1, 3, 1, 2, 1])
        self.assertTrue(np.max(np.abs(result1 - expected_result1)) < 1e-6)


class TestFindAnglePairs(unittest.TestCase):

    def test_correct_result_simple(self):
        edi1 = np.array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
        edi2 = np.array([[0, 1], [0, 2], [1, 0], [2, 0]])

        self.assertTrue(np.all(get_angle_indices(edi2)[1] == np.array([[1, 0, 2], [2, 0, 1]])))
        self.assertTrue(np.all(get_angle_indices(edi1)[1] == np.array([[0, 1, 2],
                                                                       [0, 2, 1],
                                                                       [1, 0, 2],
                                                                       [1, 2, 0],
                                                                       [2, 0, 1],
                                                                       [2, 1, 0]
                                                                       ])))
        # print(get_angle_indices(edi1))

    def test_matching_indices(self):
        edi = np.array([[0, 1], [1, 0], [1, 6], [2, 3], [3, 2], [3, 5], [3, 7], [4, 7], [5, 3], [6, 1], [6, 7],
                        [7, 3], [7, 4], [7, 6]])
        edi_new, ind_ijk, ind_nm = get_angle_indices(edi)
        # print(edi_new[ind_nm[:, 1]][: ,:2])
        self.assertTrue(np.all(edi_new[ind_nm[:, 0]][:, 1] == edi_new[ind_nm[:, 1]][:, 0]))
        self.assertTrue(
            np.all(np.concatenate([edi_new[ind_nm[:, 0]][:, :2], edi_new[ind_nm[:, 1]][:, 1:]], axis=-1) == ind_ijk))


class TestAngleCompute(unittest.TestCase):

    def test_get_angle(self):
        coord = np.array([[0, 0, 0], [0, 0 , 1], [0, 1, 0]])
        indices = np.array([[1, 0, 2], [2, 0, 1]])
        result = get_angle(coord, indices)/2/np.pi*360
        expected_result = np.array([[90], [90]])
        # print(result)
        self.assertTrue(np.max(np.abs(result - expected_result)) < 1e-6)


class TestAngleEdge(unittest.TestCase):

    def test_get_angle(self):
        coord = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        edge_idx = np.array([[0, 1], [0, 2]])
        angle_idx = np.array([[0, 1]])
        result = get_angle_between_edges(coord, edge_idx, angle_idx)/2/np.pi*360
        expected_result = np.array([[90]])
        # print(result)
        self.assertTrue(np.max(np.abs(result - expected_result)) < 1e-6)


if __name__ == '__main__':
    unittest.main()
