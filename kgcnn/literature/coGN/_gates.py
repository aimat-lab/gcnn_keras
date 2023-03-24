import tensorflow as tf
import tensorflow.keras as ks


class HadamardProductGate(ks.layers.Layer):
    """Simple gate layer that does element-wise gating, where gating values are computed with a single Dense layer."""

    def __init__(self, units, reverse_inputs=False, return_twice=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.reverse_inputs = reverse_inputs
        self.return_twice = return_twice
        self.gate_control_layer = ks.layers.Dense(
            units=self.units, activation="sigmoid"
        )

    def call(self, signal, gate_control):
        if self.reverse_inputs:
            signal, gate_control = gate_control, signal

        gate = self.gate_control_layer(gate_control)
        gated_signal = gate * signal
        if self.return_twice:
            return (gated_signal, gated_signal)
        return gated_signal
