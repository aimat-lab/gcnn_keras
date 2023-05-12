from hashlib import md5
# import logging
from pymatgen.core.structure import Structure
from typing import Callable, Union
from networkx import MultiDiGraph
from kgcnn.graph.base import GraphDict

# A separate module logger is not need for the base class.
# logging.basicConfig()  # Module logger
# module_logger = logging.getLogger(__name__)
# module_logger.setLevel(logging.INFO)


class CrystalPreprocessor(Callable[[Structure], MultiDiGraph]):
    """Base class for crystal preprocessors.

    Concrete CrystalPreprocessors must be implemented as subclasses.
    """

    node_attributes = []
    edge_attributes = []
    graph_attributes = []

    def __init__(self, output_graph_as_dict: bool = False):
        self.output_graph_as_dict = output_graph_as_dict

    def call(self, structure: Structure) -> MultiDiGraph:
        """Should be implemented in a subclass.

        Args:
            structure (Structure): Crystal for which the graph representation should be calculated.

        Raises:
            NotImplementedError:Should be implemented in a subclass.

        Returns:
            MultiDiGraph: Graph representation of the crystal.
        """
        raise NotImplementedError("Must be implemented in sub-classes.")

    def __call__(self, structure: Structure) -> Union[MultiDiGraph, GraphDict]:
        """Should be implemented in a subclass.

        Args:
            structure (Structure): Crystal for which the graph representation should be calculated.

        Raises:
            NotImplementedError:Should be implemented in a subclass.

        Returns:
            MultiDiGraph: Graph representation of the crystal.
        """
        if self.output_graph_as_dict:
            nxg = self.call(structure)
            g = GraphDict()
            g.from_networkx(
                nxg, node_attributes=self.node_attributes, edge_attributes=self.edge_attributes,
                graph_attributes=self.graph_attributes, reverse_edge_indices=True)
            return g
        return self.call(structure)

    def get_config(self) -> dict:
        """Returns a dictionary uniquely identifying the CrystalPreprocessor and its configuration.

        Returns:
            dict: A dictionary uniquely identifying the CrystalPreprocessor and its configuration.
        """
        config = vars(self)
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        config['preprocessor'] = self.__class__.__name__
        return config

    def hash(self) -> str:
        """Generates a unique hash for the CrystalPreprocessor and its configuration.

        Returns:
            str: A unique hash for the CrystalPreprocessor and its configuration.
        """
        return md5(str(self.get_config()).encode()).hexdigest()

    def __hash__(self):
        return int(self.hash(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)
