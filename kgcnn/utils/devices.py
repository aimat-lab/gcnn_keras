import tensorflow as tf
import logging
from typing import Union


logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def set_devices_gpu(gpu_ids_list: Union[list, int], restrict_memory: bool = True):
    r"""Set the visible devices from a list of GPUs. Used to assign a process to a separate GPU.

    .. note::

        Important is to restrict memory growth since a single tensorflow process will allocate almost all
        GPU memory, so for example two fits can not run on same GPU.

    Args:
        gpu_ids_list (list): Device list.
        restrict_memory (bool): Whether to restrict memory growth. Default is True.

    Returns:
        None.
    """
    if gpu_ids_list is None:
        return

    if isinstance(gpu_ids_list, int):
        gpu_ids_list = [gpu_ids_list]

    if len(gpu_ids_list) <= 0:
        module_logger.info("No gpu to set")
        return

    if tf.test.is_built_with_gpu_support() is False and tf.test.is_built_with_cuda() is False:
        module_logger.warning("No cuda support")
        module_logger.warning("Can not set GPU")
        return

    try:
        gpus = tf.config.list_physical_devices('GPU')
    except:
        module_logger.error("Can not get device list, do nothing.")
        return

    if isinstance(gpus, list):
        if len(gpus) <= 0:
            module_logger.warning("No devices found")
            module_logger.warning("Can not set GPU")
            return
        try:
            gpus_used = [gpus[i] for i in gpu_ids_list if 0 <= i < len(gpus)]
            tf.config.set_visible_devices(gpus_used, 'GPU')
            module_logger.info("Setting visible devices: %s" % gpus_used)
            if restrict_memory:
                for gpu in gpus_used:
                    module_logger.info("Restrict Memory: %s" % gpu.name)
                    tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            module_logger.info("Physical GPUS: %s, Logical GPUS: %s" % (len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
