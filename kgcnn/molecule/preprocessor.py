import numpy as np
from kgcnn.molecule.graph_rdkit import MolecularGraphRDKit
from kgcnn.graph.base import GraphPreProcessorBase
from kgcnn.molecule.methods import inverse_global_proton_dict
from kgcnn.molecule.io import parse_list_to_xyz_str


_mol_graph_interface = MolecularGraphRDKit


class SetMolBondIndices(GraphPreProcessorBase):
    """

    Args:
        node_coordinates:
        node_symbol:
        node_number:
        edge_indices:
        edge_number:
        name:
    """

    def __init__(self, *, node_coordinates: str = "node_coordinates", node_symbol: str = "node_symbol",
                 node_number: str = "node_number",
                 edge_indices: str = "edge_indices", edge_number: str = "edge_number",
                 name="set_mol_bond_indices", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_coordinates": node_coordinates, "node_number": node_number,
                                "node_symbol": node_symbol})
        self._to_assign = [edge_indices, edge_number]
        self._config_kwargs.update({
            "edge_indices": edge_indices, "node_coordinates": node_coordinates, "node_number": node_number,
            "node_symbol": node_symbol, "edge_number": edge_number})

    def call(self, node_coordinates: np.ndarray, node_symbol: np.ndarray, node_number: np.ndarray):
        if node_symbol is None:
            node_symbol = [inverse_global_proton_dict(int(x)) for x in node_number]
        else:
            node_symbol = [str(x) for x in node_symbol]
        mol = _mol_graph_interface()
        mol = mol.from_xyz(parse_list_to_xyz_str([node_symbol, node_coordinates.tolist()], number_coordinates=3))
        if mol.mol is None:
            return None, None
        idx, edge_num = mol.edge_number
        return idx, edge_num


class SetAttributes(GraphPreProcessorBase):
    pass
