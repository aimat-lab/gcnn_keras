import ase
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from kgcnn.graph.base import GraphDict, GraphPreProcessorBase
from kgcnn.graph.serial import get_preprocessor
from kgcnn.data.base import MemoryGraphList
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


class KgcnnCalculator(ase.calculators.calculator.Calculator):
    r"""ASE calculator for machine learning models from :obj:`kgcnn`."""

    def __init__(self,
                 models: list = None,
                 model_inputs: Union[list, dict] = None,
                 model_outputs: Union[list, dict] = None,
                 converter: AtomsToGraphConverter = None,
                 graph_preprocessors: List[dict] = None,
                 scaler: list = None,
                 **kwargs):
        super(KgcnnCalculator, self).__init__(**kwargs)

        self.converter = converter
        self.models = models
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs
        self.graph_preprocessors = graph_preprocessors
        self.scaler = scaler

    def _model_load(self, file_path: Union[List[str], str]) -> list:
        pass

    @staticmethod
    def _translate_properties(properties, translation) -> dict:
        if isinstance(translation, list):
            assert isinstance(properties, list), "With '%s' require list for '%s'." % (translation, properties)
            output = {key: properties[i] for i, key in enumerate(translation)}
        elif isinstance(translation, dict):
            assert isinstance(properties, dict), "With '%s' require dict for '%s'." % (translation, properties)
            output = {key: properties[value] for key, value in translation.items()}
        elif isinstance(translation, str):
            assert not isinstance(properties, (list, dict)), "Must be array for str '%s'." % properties
            output = {translation: properties}
        else:
            raise TypeError("'%s' output translation must be 'str', 'dict' or 'list'." % properties)
        return output

    def _model_predict(self, atoms: Union[List[Atoms], Atoms]):
        graph_list = self.converter(atoms)  # type MemoryGraphList
        graph_list.map_list(self.graph_preprocessors)
        tensor_input = graph_list.tensor(self.model_inputs)
        outputs = {}
        for s, m in zip(self.scaler, self.models):

            try:
                tensor_output = m(tensor_input, training=False)
            except ValueError:
                tensor_output = m.predict(tensor_input)

            # Translate output
            tensor_output = self._translate_properties(tensor_output, self.model_outputs)

            # Rescale output
            if s is not None:
                if hasattr(s, "_standardize_coordinates"):
                    assert s._standardize_coordinates is False, "Scaling of model input is not supported."
                tensor_output = s.inverse_transform(tensor_output)

            # Prepare output.

    # Interface to ASE calculator scheme.
    def calculate(self, atoms=None, properties=None, system_changes=None):

        if not self.calculation_required(atoms, properties):
            # Nothing to do.
            return
