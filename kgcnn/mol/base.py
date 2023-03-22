import logging
import numpy as np

# Module logger
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class MolGraphInterface:
    r"""The `MolGraphInterface` defines the base class interface to extract a molecular graph.

    The method implementation to generate a molecule-instance from smiles etc. can be obtained from different backends
    like `RDkit` . The mol-instance of a chemical informatics package like `RDkit` is treated via composition.
    The interface is designed to extract a graph from a mol instance, not to make a mol object from a graph.

    """

    def __init__(self, mol=None, make_directed: bool = False):
        """Set the mol attribute for composition. This mol instances will be the backend molecule class.

        Args:
            mol: Instance of a molecule from chemical informatics package.
            make_directed (bool): Whether the edges are directed. Default is False.

        """
        self.mol = mol
        self._make_directed = make_directed

    def add_hs(self, **kwargs):
        """Add hydrogen to molecule instance."""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def remove_hs(self, **kwargs):
        """Remove hydrogen from molecule instance."""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def make_conformer(self, **kwargs):
        """Generate a conformer guess for molecule instance."""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def optimize_conformer(self, **kwargs):
        """Optimize conformer of molecule instance."""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def from_smiles(self, smile: str, **kwargs):
        """Main method to generate a molecule from smiles string representation.

        Args:
            smile (str): Smile string representation of a molecule.

        Returns:
            self
        """
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def to_smiles(self, **kwargs):
        """Return a smile string representation of the mol instance.

        Returns:
            smile (str): Smile string.
        """
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def from_mol_block(self, mol_block: str, keep_hs: bool = True, **kwargs):
        """Set mol-instance from a more extensive string representation containing coordinates and bond information.

        Args:
            mol_block (str): Mol-block representation of a molecule.
            keep_hs (str): Whether to keep hydrogen in mol-block. Default is True.

        Returns:
            self
        """
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def to_mol_block(self, **kwargs):
        """Make a more extensive string representation containing coordinates and bond information from self.

        Returns:
            mol_block (str): Mol-block representation of a molecule.
        """
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def clean(self, **kwargs):
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def compute_partial_charges(self, method="gasteiger", **kwargs):
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def node_number(self):
        """Return list of node numbers which is the atomic number of atoms in the molecule"""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def node_symbol(self):
        """Return a list of atomic symbols of the molecule."""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def node_coordinates(self):
        """Return a list of atomic coordinates of the molecule."""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def edge_indices(self):
        """Return a list of edge indices of the molecule."""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def edge_number(self):
        """Return a list of edge number that represents the bond order."""
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def edge_attributes(self, properties: list, encoder: dict):
        """Make edge attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
        """
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def node_attributes(self, properties: list, encoder: dict):
        """Make node attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
        """
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    def graph_attributes(self, properties: list, encoder: dict):
        """Make graph attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
        """
        raise NotImplementedError("Method for `MolGraphInterface` must be implemented in sub-class.")

    @staticmethod
    def _check_encoder(encoder: dict, possible_keys: list, raise_error: bool = False):
        """Verify and check if encoder dictionary inputs is within possible properties. If a key has to be removed,
        a warning is issued.

        Args:
            encoder (dict): Dictionary of callable encoder function or class. Key matches properties.
            possible_keys (list): List of allowed keys for encoder.
            raise_error (bool): Whether to raise an error on wrong identifier.

        Returns:
            dict: Cleaned encoder dictionary.
        """
        if encoder is None:
            return {}
        # Check if encoder is given for unknown identifier.
        # Encoder is only intended for string-based properties.
        encoder_unknown = [x for x in encoder if x not in possible_keys]
        if len(encoder_unknown) > 0:
            msg = "Encoder property not known %s" % encoder_unknown
            if raise_error:
                module_logger.error(msg)
                raise ValueError(msg)
            else:
                module_logger.warning(msg)
        encoder = {key: value for key, value in encoder.items() if key not in encoder_unknown}
        return encoder

    @staticmethod
    def _check_properties_list(properties: list,
                               possible_properties: list,
                               attribute_name: str,
                               raise_error: bool = False):
        """Verify and check if list of string identifier match expected properties. If an identifier has to be removed,
        a warning is issued. Non-string properties i.e. class or functions to extract properties are ignored.

        Args:
            properties (list): List of requested string identifier. Key matches properties.
            possible_properties (list): List of allowed string identifier for properties.
            attribute_name(str): A name for the properties. E.g. bond, node or graph.
            raise_error (bool): Whether to raise an error on wrong identifier.

        Returns:
            dict: Cleaned encoder dictionary.
        """
        if properties is None:
            return []
        # Check if string identifier match the list of possible keys.
        props_unknown = []
        for x in properties:
            if isinstance(x, str):
                if x not in possible_properties:
                    props_unknown.append(x)
        if len(props_unknown) > 0:
            msg = "%s properties are not defined, ignore following keys: %s" % (attribute_name, props_unknown)
            if raise_error:
                module_logger.error(msg)
                raise ValueError(msg)
            else:
                module_logger.warning(msg)
        props = []
        for x in properties:
            if isinstance(x, str):
                if x in possible_properties:
                    props.append(x)
            else:
                props.append(x)
        return props

    @staticmethod
    def _sort_bonds(bond_idx, bond_info=None):
        # Sort directed bonds
        bond_idx = np.array(bond_idx, dtype="int64")
        bonds1, bonds2 = None, None
        if len(bond_idx) > 0:
            order1 = np.argsort(bond_idx[:, 1], axis=0, kind='mergesort')  # stable!
            ind1 = bond_idx[order1]
            if bond_info:
                bonds1 = [bond_info[i] for i in order1]
            order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')  # stable!
            ind2 = ind1[order2]
            if bond_info:
                bonds2 = [bonds1[i] for i in order2]
            # Take the sorted bonds
            bond_idx = ind2
            bond_info = bonds2
        return bond_idx, bond_info
