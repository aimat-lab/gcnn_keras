from keras_core.backend import backend

# Import backend functions.
if backend() == "tensorflow":
    from kgcnn.backend.tensorflow import *
elif backend() == "jax":
    from kgcnn.backend.jax import *
elif backend() == "torch":
    from kgcnn.backend.torch import *
elif backend() == "numpy":
    from kgcnn.backend.numpy import *
else:
    raise ValueError(f"Unable to import backend : {backend()}")
