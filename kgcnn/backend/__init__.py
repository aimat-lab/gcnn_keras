import tree
from keras import KerasTensor
from keras.backend import backend


def any_symbolic_tensors(args=None, kwargs=None):
    args = args or ()
    kwargs = kwargs or {}
    for x in tree.flatten((args, kwargs)):
        if isinstance(x, KerasTensor):
            return True
    return False


# Import backend functions.
if backend() == "tensorflow":
    from kgcnn.backend._tensorflow import *
elif backend() == "jax":
    from kgcnn.backend._jax import *
elif backend() == "torch":
    from kgcnn.backend._torch import *
elif backend() == "numpy":
    from kgcnn.backend._numpy import *
else:
    raise ValueError(f"Unable to import backend : {backend()}")
