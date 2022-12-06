import numpy as np
from kgcnn.utils.serial import serialize, deserialize
from kgcnn.graph.base import GraphPostProcessorBase, GraphDict


class ExtensiveEnergyForceScalerPostprocessor(GraphPostProcessorBase):

    def __init__(self, scaler, energy: str = "energy",
                 force: str = "forces", atomic_number: str = "node_number",
                 coordinates: str = "node_coordinates",
                 name="extensive_energy_force_scaler", **kwargs):
        super(ExtensiveEnergyForceScalerPostprocessor, self).__init__(name=name, **kwargs)
        if isinstance(scaler, dict):
            self.scaler = deserialize(scaler)
        else:
            self.scaler = scaler
        self._to_obtain_pre = {"X": coordinates, "atomic_number": atomic_number}
        self._to_obtain = {"y": energy, "force": force}
        self._to_assign = [energy, force]
        self._config_kwargs.update(
            {"scaler": serialize(self.scaler), "energy": energy, "force": force, "atomic_number": atomic_number,
             "coordinates": coordinates})

    def call(self, X, y, force, atomic_number):
        # Need batch with one energy etc.
        _, energy, forces = self.scaler.inverse_transform(
            X=[X], y=np.array([y]), force=[force], atomic_number=[atomic_number])
        return energy[0], forces[0]
