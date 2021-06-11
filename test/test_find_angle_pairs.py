import tensorflow as tf
import numpy as np
import unittest

from kgcnn.data.mol.methods import get_angle_indices

class TestFindAnglePairs(unittest.TestCase):


    def test_correct_result_simple(self):
        edi1 = np.array([[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]])
        edi2 = np.array([[0,1],[0,2],[1,0],[2,0]])

        self.assertTrue(np.all(get_angle_indices(edi2)[1] == np.array([[1,0,2],[2,0,1]])))
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
                         [7, 3], [7, 4],[7, 6]])
        edi_new, ind_ijk, ind_nm = get_angle_indices(edi)
        # print(edi_new[ind_nm[:, 1]][: ,:2])
        self.assertTrue(np.all(edi_new[ind_nm[:,0]][:,1] == edi_new[ind_nm[:,1]][:,0]))
        self.assertTrue(np.all(np.concatenate([edi_new[ind_nm[:, 0]][:, :2], edi_new[ind_nm[:, 1]][:,1:]],axis=-1) == ind_ijk))




if __name__ == '__main__':
    unittest.main()


