import numpy as np
from kgcnn.utils.serial import serialize, deserialize
from kgcnn.graph.base import GraphPostProcessorBase, GraphDict


class ExtensiveEnergyForceScalerPostprocessor(GraphPostProcessorBase):

    def __init__(self, scaler, energy: str = "energy",
                 force: str = "forces", atomic_number: str = "node_number",
                 coordinates: str = "node_coordinates", **kwargs):
        super(ExtensiveEnergyForceScalerPostprocessor, self).__init__(**kwargs)
        if isinstance(scaler, dict):
            self.scaler = deserialize(scaler)
        else:
            self.scaler = scaler
        self._to_obtain_x = {"X": coordinates, "atomic_number": atomic_number}
        self._to_obtain_y = {"y": energy, "force": force}
        self._to_assign_y = [energy, force]
        self._config_kwargs.update(
            {"scaler": serialize(self.scaler), "energy": energy, "force": force, "atomic_number": atomic_number,
             "coordinates": coordinates})

    def call(self, **kwargs):
        _, energy, forces = self.scaler.inverse_transform(**kwargs)
        return energy, forces
