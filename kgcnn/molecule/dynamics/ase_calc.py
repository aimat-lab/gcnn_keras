import ase
import tensorflow as tf
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from kgcnn.graph.base import GraphDict
from kgcnn.data.base import MemoryGraphList
from copy import deepcopy
from typing import Union, List

ks = tf.keras


class AtomsToGraphConverter:
    r"""Convert :obj:`ase.Atoms` object to a :obj:`GraphDict` dictionary.

    Simple tool to get named properties from :obj:`ase.Atoms`. Note that the actual graph indices and connections
    have to be generated with :obj:`GraphPreProcessorBase` instances.

    Example usage:

     .. code-block:: python

        import numpy as np
        from ase import Atoms
        from kgcnn.md.ase_calc import AtomsToGraphConverter
        atoms = Atoms('N2', positions=[[0, 0, -1], [0, 0, 1]])
        trans = AtomsToGraphConverter({
            "node_number": "get_atomic_numbers", "node_coordinates": "get_positions",
            "node_symbol": "get_chemical_symbols"})
        print(trans(atoms))

    """

    def __init__(self, properties: dict = None):
        r"""Set up :obj:`AtomsToGraphConverter` converter.

        Args:
            properties (dict): Dictionary of graph properties linked to :obj:`ase.Atoms` get attribute methods.
                Default is {"node_number": "get_atomic_numbers", "node_coordinates": "get_positions",
                "node_symbol": "get_chemical_symbols"}.
        """
        if properties is None:
            properties = {"node_number": "get_atomic_numbers", "node_coordinates": "get_positions",
                          "node_symbol": "get_chemical_symbols"}
        self.properties = deepcopy(properties)

    def __call__(self, atoms: Union[List[Atoms], Atoms]) -> MemoryGraphList:
        r"""Make :obj:`GraphDict` objects from :obj:`ase.Atoms`.

        Args:
            atoms (list): List of :obj:`ase.Atoms` objects or single ASE atoms object.

        Returns:
            MemoryGraphList: List of :obj:`GraphDict` objects.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        graph_list = []
        for i, x in enumerate(atoms):
            g = GraphDict()
            for key, value in self.properties.items():
                g.update({key: np.array(getattr(Atoms, value)(x))})
            graph_list.append(g)

        return MemoryGraphList(graph_list)

    def get_config(self):
        """Get config for this class."""
        config = {"properties": self.properties}
        return config


class KgcnnSingleCalculator(ase.calculators.calculator.Calculator):
    r"""ASE calculator for machine learning models from :obj:`kgcnn`."""

    implemented_properties = ["energy", "forces"]

    def __init__(self,
                 model_predictor=None,
                 atoms_converter: AtomsToGraphConverter = None,
                 squeeze_energy: bool = True,
                 **kwargs):
        super(KgcnnSingleCalculator, self).__init__(**kwargs)
        self.model_predictor = model_predictor
        self.atoms_converter = atoms_converter
        self.squeeze_energy = squeeze_energy

    # Interface to ASE calculator scheme.
    def calculate(self, atoms=None, properties=None, system_changes=None):

        if not self.calculation_required(atoms, properties):
            # Nothing to do.
            return
        super(KgcnnSingleCalculator, self).calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        graph_list = self.atoms_converter(atoms)
        output_dict = self.model_predictor(graph_list)

        # Update.
        assert len(output_dict) == 1, "ASE Calculator updates only one structure for now."
        self.results.update(output_dict[0].to_dict())
        if self.squeeze_energy:
            # For single energy only.
            if len(self.results["energy"].shape) > 0:
                self.results["energy"] = np.squeeze(self.results["energy"])
