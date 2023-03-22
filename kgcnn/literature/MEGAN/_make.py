import tensorflow as tf
from typing import Optional
from ._model import MEGAN
from ._model import __model_version__

ks = tf.keras


def make_model(inputs: Optional[list] = None,
               **kwargs
               ):
    r"""Functional model definition of MEGAN. Please check documentation of :obj:`kgcnn.literature.MEGAN` .

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        kwargs: Kwargs for MEGAN model. Please check documentation of :obj:`kgcnn.literature.MEGAN` .

    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Building the actual model
    megan = MEGAN(**kwargs)

    # Wrapping the actual model inside a keras functional model to be able to account for the input shapes
    # definitions which are provided.
    layer_inputs = [ks.layers.Input(**kwargs) for kwargs in inputs]
    outputs = megan(layer_inputs)
    model = ks.models.Model(inputs=layer_inputs, outputs=outputs)

    model.__kgcnn_model_version__ = __model_version__
    return model
