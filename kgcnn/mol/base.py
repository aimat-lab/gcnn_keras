class MolGraphInterface:
    r"""The `MolGraphInterface` defines the base class interface to handle a molecular graph. The method implementation
    to generate a mol-instance from smiles etc. can be obtained from different backends like `rdkit`. The mol-instance
    of a chemical informatics package like `rdkit` is treated via composition. The interface is designed to
    extract a graph from a mol instance not to make a mol object from a graph, but could be extended that way.

    """

    def __init__(self, mol=None, add_hydrogen: bool = False):
        """Set the mol attribute for composition. This mol instances will be the backends molecule class.

        Args:
            mol: Instance of a molecule from chemical informatics package.
            add_hydrogen (bool): Whether to add or ignore hydrogen in the molecule.
        """
        self.mol = mol
        self._add_hydrogen = add_hydrogen

    def from_smiles(self, smile: str, **kwargs):
        """Main method to generate a molecule from smiles string representation.

        Args:
            smile (str): Smile string representation of a molecule.

        Returns:
            self
        """
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    def to_smiles(self):
        """Return a smile string representation of the mol instance.

        Returns:
            smile (str): Smile string.
        """
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    def from_mol_block(self, mol_block: str):
        """Set mol-instance from a more extensive string representation containing coordinates and bond information.

        Args:
            mol_block (str): Mol-block representation of a molecule.

        Returns:
            self
        """
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    def to_mol_block(self):
        """Make a more extensive string representation containing coordinates and bond information from self.

        Returns:
            mol_block (str): Mol-block representation of a molecule.
        """
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def node_number(self):
        """Return list of node numbers which is the atomic number of atoms in the molecule"""
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def node_symbol(self):
        """Return a list of atomic symbols of the molecule."""
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def node_coordinates(self):
        """Return a list of atomic coordinates of the molecule."""
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def edge_indices(self):
        """Return a list of edge indices of the molecule."""
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    @property
    def edge_number(self):
        """Return a list of edge number that represents the bond order."""
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    def edge_attributes(self, properties: list, encoder: dict):
        """Make edge attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
        """
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    def node_attributes(self, properties: list, encoder: dict):
        """Make node attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
        """
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    def graph_attributes(self, properties: list, encoder: dict):
        """Make graph attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
        """
        raise NotImplementedError("ERROR:kgcnn: Method for `MolGraphInterface` must be implemented in sub-class.")

    @staticmethod
    def _check_encoder(encoder: dict, possible_keys: list):
        """Verify and check if encoder dictionary inputs is within possible properties. If a key has to be removed,
        a warning is issued.

        Args:
            encoder (dict): Dictionary of callable encoder function or class. Key matches properties.
            possible_keys (list): List of allowed keys for encoder.

        Returns:
            dict: Cleaned encoder dictionary.
        """
        if encoder is None:
            encoder = {}
        else:
            encoder_unknown = [x for x in encoder if x not in possible_keys]
            if len(encoder_unknown) > 0:
                print("WARNING: Encoder property not known", encoder_unknown)
            encoder = {key: value for key, value in encoder.items() if key not in encoder_unknown}
        return encoder

    @staticmethod
    def _check_properties_list(properties: list, possible_properties: list, attribute_name: str):
        """Verify and check if list of string identifier match expected properties. If an identifier has to be removed,
        a warning is issued.

        Args:
            properties (list): List of requested string identifier. Key matches properties.
            possible_properties (list): List of allowed string identifier for properties.
            attribute_name(str): A name for the properties.

        Returns:
            dict: Cleaned encoder dictionary.
        """
        if properties is None:
            props = [x for x in possible_properties]
        else:
            props_unknown = [x for x in properties if x not in possible_properties]
            if len(props_unknown) > 0:
                print("WARNING:kgcnn: %s property is not defined, ignore following keys:" % attribute_name,
                      props_unknown)
            props = [x for x in properties if x in possible_properties]
        return props