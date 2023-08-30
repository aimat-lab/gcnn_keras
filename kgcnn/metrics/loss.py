from kgcnn.losses import *
import logging

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

module_logger.error("Module `kgcnn.metrics.loss` is deprecated, please move to `kgcnn.losses` .")