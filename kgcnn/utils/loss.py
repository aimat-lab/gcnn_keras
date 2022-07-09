import tensorflow as tf
import logging
from kgcnn.metrics.metrics import *

# For old reference. Moved to kgcnn.training module.
logging.error(
    "Module '%s' is deprecated and will be removed in future versions. Please move to 'kgcnn.metrics.metrics'." % __name__)
