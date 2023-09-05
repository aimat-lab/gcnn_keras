from keras_core.backend import backend

# Import backend functions.
if backend() == "tensorflow":
    from kgcnn.backend.tensorflow import (scatter_max, scatter_min, scatter_mean)
elif backend() == "jax":
    from kgcnn.backend.tensorflow import (scatter_max, scatter_min, scatter_mean)
elif backend() == "torch":
    from kgcnn.backend.tensorflow import (scatter_max, scatter_min, scatter_mean)
elif backend() == "numpy":
    from kgcnn.backend.tensorflow import (scatter_max, scatter_min, scatter_mean)
else:
    raise ValueError(f"Unable to import backend : {backend()}")
