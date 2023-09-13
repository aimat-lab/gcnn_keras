from keras_core.backend import backend
import logging

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def check_device():

    if backend() == "tensorflow":
        import tensorflow as tf
        cuda_is_available = tf.test.is_built_with_gpu_support() and tf.test.is_built_with_cuda()
        physical_device_list = tf.config.list_physical_devices()
        if physical_device_list:
            physical_device_name = [
                tf.config.experimental.get_device_details(x) for x in physical_device_list]
        else:
            physical_device_name = []
        logical_device_list = tf.config.experimental.list_logical_devices()
        if physical_device_name:
            try:
                memory_info = [tf.config.experimental.get_memory_info(x.name) for x in physical_device_list]
            except (TypeError, ValueError):
                memory_info = []
        else:
            memory_info = []

    elif backend() == "torch":
        import torch
        cuda_is_available = torch.cuda.is_available()
        physical_device_name = [torch.cuda.get_device_name(x) for x in range(torch.cuda.device_count())]
        logical_device_list = [x for x in range(torch.cuda.device_count())]
        memory_info = [{"allocated": round(torch.cuda.memory_allocated(i)/1024**3, 1),
                        "cached": round(torch.cuda.memory_reserved(i)/1024**3, 1)} for i in logical_device_list]
    elif backend() == "jax":
        import jax
        jax_devices = jax.devices()
        cuda_is_available = any([x.device_kind == "gpu" for x in jax_devices])
        physical_device_name = [x for x in jax_devices]
        logical_device_list = [x.id for x in jax_devices]
        memory_info = [x.memory_stats() for x in jax_devices]

    else:
        raise NotImplementedError("Backend %s is not supported for `check_device_cuda` .")

    out_info = {
        "cuda_available": "%s" % cuda_is_available,
        "device_name": "%s" % physical_device_name,
        "device_id": "%s" % logical_device_list,
        "device_memory": "%s" % memory_info,
    }
    module_logger.info("GPU: Device information: %s." % out_info)
    return out_info
