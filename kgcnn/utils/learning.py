import numpy as np
import tensorflow as tf
import logging
from kgcnn.training.scheduler import *
from kgcnn.training.schedule import *
from kgcnn.training.callbacks import *


# For old reference. Moved to kgcnn.training module.
logging.error(
    "Module '%s' is deprecated and will be removed in future versions. Please move to 'kgcnn.training'." % __name__)