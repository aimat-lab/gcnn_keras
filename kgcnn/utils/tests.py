from typing import Union
import unittest
import numpy as np
from keras import ops


def compare_static_shapes(found: Union[tuple, None], expected: Union[tuple, None]):
    if found is None and expected is None:
        return True
    elif found is None and expected is not None:
        return False
    elif found is not None and expected is None:
        return True
    elif len(found) != len(expected):
        return False
    shapes_okay = []
    for f, e in zip(found, expected):
        if f is None and e is not None:
            shapes_okay.append(False)
        elif f is not None and e is None:
            shapes_okay.append(True)
        elif f is None and e is None:
            shapes_okay.append(True)
        elif f == e:
            shapes_okay.append(True)
        else:
            shapes_okay.append(False)
    return all(shapes_okay)


class TestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def assertAllClose(x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        if not isinstance(x1, np.ndarray):
            x1 = ops.convert_to_numpy(x1)
        if not isinstance(x2, np.ndarray):
            x2 = ops.convert_to_numpy(x2)
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol)

    def assertNotAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        try:
            self.assertAllClose(x1, x2, atol=atol, rtol=rtol, msg=msg)
        except AssertionError:
            return
        msg = msg or ""
        raise AssertionError(
            f"The two values are close at all elements. \n"
            f"{msg}.\n"
            f"Values: {x1}"
        )

    def assertAllEqual(self, x1, x2, msg=None):
        self.assertEqual(len(x1), len(x2), msg=msg)
        for e1, e2 in zip(x1, x2):
            if isinstance(e1, (list, tuple)) or isinstance(e2, (list, tuple)):
                self.assertAllEqual(e1, e2, msg=msg)
            else:
                e1 = ops.convert_to_numpy(e1)
                e2 = ops.convert_to_numpy(e2)
                self.assertEqual(e1, e2, msg=msg)

    def assertLen(self, iterable, expected_len, msg=None):
        self.assertEqual(len(iterable), expected_len, msg=msg)
