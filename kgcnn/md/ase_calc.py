import ase
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from kgcnn.graph.base import GraphDict, GraphPreProcessorBase
from kgcnn.graph.serial import get_preprocessor
from copy import deepcopy
from typing import Union, List


class AtomsToGraphConverter:
    r"""Convert :obj:`ase.Atoms` object to a :obj:`GraphDict` dictionary.

    Simple tool to get named properties from :obj:`ase.Atoms`. Note that the actual graph indices and connections
    have to be generated with :obj:`GraphPreProcessorBase` instances.

    Example usage:

     .. code-block:: python

        import numpy as np
        from ase import Atoms
        from kgcnn.md.ase_calc import AtomsToGraphTransform
        atoms = Atoms('N2', positions=[[0, 0, -1], [0, 0, 1]])
        trans = AtomsToGraphTransform({
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

    def __call__(self, atoms: Union[List[Atoms], Atoms]) -> list:
        r"""Make :obj:`GraphDict` objects from :obj:`ase.Atoms`.

        Args:
            atoms (list): List of :obj:`ase.Atoms` objects or single ASE atoms object.

        Returns:
            list: List of :obj:`GraphDict` objects.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        graph_list = []
        for i, x in enumerate(atoms):
            g = GraphDict()
            for key, value in self.properties.items():
                g.update({key: np.array(getattr(Atoms, value)(x))})
            graph_list.append(g)

        return graph_list

    def get_config(self):
        """Get config for this class."""
        config = {"properties": self.properties}
        return config


class KgcnnCalculator(ase.calculators.calculator.Calculator):
    r"""ASE calculator for machine learning models from :obj:`kgcnn`."""

    def __init__(self,
                 models=None,
                 converter: AtomsToGraphConverter = None,
                 graph_preprocessors: Union[list, dict, GraphPreProcessorBase] = None,
                 scaler=None,
                 **kwargs):
        super(KgcnnCalculator, self).__init__(**kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass
