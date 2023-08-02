import numpy as np
from kgcnn.molecule.graph_rdkit import MolecularGraphRDKit
from kgcnn.graph.base import GraphPreProcessorBase
from kgcnn.molecule.methods import inverse_global_proton_dict
from kgcnn.molecule.io import parse_list_to_xyz_str
from kgcnn.molecule.encoder import OneHotEncoder
from kgcnn.utils.serial import serialize
from kgcnn.molecule.serial import deserialize_encoder

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


class SetMolAttributes(GraphPreProcessorBase):
    """

    .. code-block:: python

        from kgcnn.data.datasets.QM7Dataset import QM7Dataset
        from kgcnn.molecule.preprocessor import SetMolAttributes
        ds = QM7Dataset()
        pp = SetMolAttributes()
        print(pp(ds[0]))

    """

    _default_node_attributes = [
        'Symbol', 'TotalDegree', 'FormalCharge', 'NumRadicalElectrons', 'Hybridization',
        'IsAromatic', 'IsInRing', 'TotalNumHs', 'CIPCode', "ChiralityPossible", "ChiralTag"
    ]
    _default_node_encoders = {
        'Symbol': OneHotEncoder(
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
            dtype="str"
        ),
        'Hybridization': OneHotEncoder([2, 3, 4, 5, 6]),
        'TotalDegree': OneHotEncoder([0, 1, 2, 3, 4, 5], add_unknown=False),
        'TotalNumHs': OneHotEncoder([0, 1, 2, 3, 4], add_unknown=False),
        'CIPCode': OneHotEncoder(['R', 'S'], add_unknown=False, dtype='str'),
        "ChiralityPossible": OneHotEncoder(["1"], add_unknown=False, dtype='str'),
    }
    _default_edge_attributes = ['BondType', 'IsAromatic', 'IsConjugated', 'IsInRing', 'Stereo']
    _default_edge_encoders = {
        'BondType': OneHotEncoder([1, 2, 3, 12], add_unknown=False),
        'Stereo': OneHotEncoder([0, 1, 2, 3], add_unknown=False)
    }
    _default_graph_attributes = ['ExactMolWt', 'NumAtoms']
    _default_graph_encoders = {}

    def __init__(self, *,
                 nodes: list = None, edges: list = None, graph: list = None,
                 encoder_nodes: dict = None,
                 encoder_edges: dict = None,
                 encoder_graph: dict = None,
                 node_coordinates: str = "node_coordinates", node_symbol: str = "node_symbol",
                 node_number: str = "node_number",
                 edge_indices: str = "edge_indices", edge_number: str = "edge_number",
                 node_attributes: str = "node_attributes", edge_attributes: str = "edge_attributes",
                 graph_attributes: str = "graph_attributes",
                 name="set_mol_attributes", **kwargs):
        super().__init__(name=name, **kwargs)
        nodes = nodes if nodes is not None else self._default_node_attributes
        edges = edges if edges is not None else self._default_edge_attributes
        graph = graph if graph is not None else self._default_graph_attributes
        encoder_nodes = encoder_nodes if encoder_nodes is not None else self._default_node_encoders
        encoder_edges = encoder_edges if encoder_edges is not None else self._default_edge_encoders
        encoder_graph = encoder_graph if encoder_graph is not None else self._default_graph_encoders

        self._to_obtain.update({"node_coordinates": node_coordinates, "node_number": node_number,
                                "node_symbol": node_symbol, "edge_indices": edge_indices, "edge_number": edge_number})
        self._to_assign = [node_attributes, edge_attributes, graph_attributes, edge_indices, edge_number]
        self._call_kwargs = {
            "nodes": nodes,
            "edges": edges,
            "graph": graph,
            "encoder_nodes": {key: deserialize_encoder(value) for key, value in encoder_nodes.items()},
            "encoder_edges": {key: deserialize_encoder(value) for key, value in encoder_edges.items()},
            "encoder_graph": {key: deserialize_encoder(value) for key, value in encoder_graph.items()}
        }
        self._config_kwargs.update({
            "edge_indices": edge_indices, "node_coordinates": node_coordinates, "node_number": node_number,
            "node_symbol": node_symbol, "edge_number": edge_number,
            "node_attributes": node_attributes, "edge_attributes": edge_attributes,
            "graph_attributes": graph_attributes,
            "nodes": nodes,
            "edges": edges,
            "graph": graph,
            "encoder_nodes": {key: serialize(value) for key, value in encoder_nodes.items()},
            "encoder_edges": {key: serialize(value) for key, value in encoder_edges.items()},
            "encoder_graph": {key: serialize(value) for key, value in encoder_graph.items()}
        })

    def call(self,
             nodes: list, edges: list, graph: list,
             encoder_nodes: dict,
             encoder_edges: dict,
             encoder_graph: dict,
             node_coordinates: np.ndarray, node_symbol: np.ndarray, node_number: np.ndarray,
             edge_indices: np.ndarray, edge_number: np.ndarray):
        if node_symbol is None:
            node_symbol = [inverse_global_proton_dict(int(x)) for x in node_number]
        else:
            node_symbol = [str(x) for x in node_symbol]
        mol = _mol_graph_interface()
        mol.from_list(node_symbol, edge_indices, edge_number, conformer=node_coordinates)
        n_att = mol.node_attributes(nodes, encoder=encoder_nodes)
        _, e_att = mol.edge_attributes(edges, encoder=encoder_edges)
        g_att = mol.graph_attributes(graph, encoder=encoder_graph)
        idx, en = mol.edge_number
        return n_att, e_att, g_att, idx, en
