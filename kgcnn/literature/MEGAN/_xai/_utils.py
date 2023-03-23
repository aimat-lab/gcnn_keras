import typing as t

import numpy as np


def flatten_importances_list(importances_list: t.List[np.ndarray],
                             ) -> t.List[float]:
    """
    Given an ``importances_list`` which is a list of numpy arrays (which may be different shapes), this
    function will flatten all the numpy arrays contained within the list and concatenate all the flattened
    arrays into a single long list of values.

    Args:
        importances_list: A list of numpy arrays which may have varying shapes.

    Returns:
        A list of all the values within the given arrays, flattened and concatenated in the order given.
    """
    flattened = []
    for importances in importances_list:
        flattened += importances.flatten().tolist()

    return flattened
