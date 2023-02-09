import numpy as np
from kgcnn.utils.serial import serialize, deserialize
from kgcnn.graph.base import GraphPostProcessorBase, GraphDict


class ExtensiveEnergyForceScalerPostprocessor(GraphPostProcessorBase):
    r"""Postprocessor to inverse-transform energies and forces from a graph dictionary.
    Note that this postprocessor requires the input- or pre-graph in call method as well.

    Args:
        scaler: Fitted instance of :obj:`ExtensiveEnergyForceScaler` .
        energy (str): Name of energy property in :obj:`graph` dict. Default is 'energy'.
        force (str): Name of force property in :obj:`graph` dict. Default is 'forces'.
        atomic_number (str): Name of atomic_number property in :obj:`pre_graph` dict. Default is 'node_number'.
            Must be given in additional :obj:`pre_graph` that is passed to call.
    """

    def __init__(self, scaler, energy: str = "energy",
                 force: str = "forces", atomic_number: str = "node_number",
                 name="extensive_energy_force_scaler", **kwargs):
        super(ExtensiveEnergyForceScalerPostprocessor, self).__init__(name=name, **kwargs)
        if isinstance(scaler, dict):
            self.scaler = deserialize(scaler)
        else:
            self.scaler = scaler
        self._to_obtain_pre = {"atomic_number": atomic_number}
        self._to_obtain = {"y": energy, "force": force}
        self._to_assign = [energy, force]
        self._config_kwargs.update(
            {"scaler": serialize(self.scaler), "energy": energy, "force": force, "atomic_number": atomic_number})

    def call(self, y, force, atomic_number):
        # Need batch with one energy etc.
        # Can copy data.
        energy, forces = self.scaler.inverse_transform(
            X=None, y=np.array([y]), force=[force], atomic_number=[atomic_number])
        return energy[0], forces[0]
