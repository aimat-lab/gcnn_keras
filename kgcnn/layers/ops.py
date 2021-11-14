import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.axis import get_positive_axis


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ExpandDims')
class ExpandDims(GraphBaseLayer):

    def __init__(self,
                 axis=-1,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis  # We do not change the axis here

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras layer."""
        if isinstance(inputs, tf.RaggedTensor):
            axis = get_positive_axis(self.axis, inputs.shape.rank + 1)
            if axis > 1 and inputs.ragged_rank == 1:
                value_tensor = inputs.values  # will be Tensor
                out_tensor = tf.expand_dims(value_tensor, axis=axis - 1)
                return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting keras call... ")
        # Try normal operaton
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(GraphBaseLayer, self).get_config()
        config.update({"axis": self.axis})
        return config