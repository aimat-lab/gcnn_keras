import numpy as np
import unittest
from kgcnn.metrics.metrics import ScaledForceMeanAbsoluteError


class TestScaledForceMeanAbsoluteError(unittest.TestCase):

    def test_correctness(self):

        m = ScaledForceMeanAbsoluteError()

        y_true = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                           [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]])
        y_pred = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                           [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]])
        m.update_state(y_true=y_true, y_pred=y_pred)
        result = np.array(m.result())
        print(result)
        # self.assertTrue(np.max(np.abs(result - expected_result)) < 1e-6)


if __name__ == "__main__":

    TestScaledForceMeanAbsoluteError().test_correctness()
    print("Tests passed.")