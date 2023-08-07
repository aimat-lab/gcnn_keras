from kgcnn.layers.aggr import *
import logging

# Module logger
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

module_logger.warning("Module `pooling` has been renamed `aggr` for better compatibility with other libraries. Please change module name.")
