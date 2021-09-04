import numpy as np

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.Descriptors

# For this module please install rdkit. See: https://www.rdkit.org/docs/Install.html
# or check https://pypi.org/project/rdkit-pypi/ via `pip install rdkit-pypi`
# or try `conda install -c rdkit rdkit` or `conda install -c conda-forge rdkit`


class OneHotEncoder:
    """Simple One-Hot-Encoding for python lists. Uses a list of possible values for a one-hot encoding of a single
    value. The translated values must support ``__eq__()`` operator. The list of possible values must be set beforehand.
    Is used as a basic encoder example for ``MolecularGraphRDKit``.
    """

    def __init__(self, one_hot_values: list, add_others: bool = True):
        """Initialize the encoder beforehand with a set of all possible values to encounter.

        Args:
            one_hot_values (list): List of possible values, matching the one-hot encoding.
            add_others (bool): Whether to add a unknown bit. Default is True.
        """
        self.one_hot_values = one_hot_values
        self.found_values = []
        self.add_others = add_others

    def __call__(self, value, **kwargs):
        r"""Encode a single value mapping it to a one-hot python list. E.g. `[0, 0, 1, 0]`

        Args:
            value: Any object that can be compared to items in ``self.one_hot_values``.
            **kwargs: Additional kwargs. Not used atm.

        Returns:
            list: Python list with 1 at value match. E.g. `[0, 0, 1, 0]`
        """
        encoded_list = [1 if x == value else 0 for x in self.one_hot_values]
        if self.add_others:
            if value not in self.one_hot_values:
                encoded_list += [1]
            else:
                encoded_list += [0]
        if value not in self.found_values:
            self.found_values += [value]
        return encoded_list


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


class MolecularGraphRDKit(MolGraphInterface):
    """A graph object representing a strict molecular graph, e.g. only chemical bonds."""

    atom_fun_dict = {
        "AtomicNum": rdkit.Chem.rdchem.Atom.GetAtomicNum,
        "Symbol": rdkit.Chem.rdchem.Atom.GetSymbol,
        "NumExplicitHs": rdkit.Chem.rdchem.Atom.GetNumExplicitHs,
        "NumImplicitHs": rdkit.Chem.rdchem.Atom.GetNumImplicitHs,
        "TotalNumHs": rdkit.Chem.rdchem.Atom.GetTotalNumHs,
        "IsAromatic": rdkit.Chem.rdchem.Atom.GetIsAromatic,
        "TotalDegree": rdkit.Chem.rdchem.Atom.GetTotalDegree,
        "TotalValence": rdkit.Chem.rdchem.Atom.GetTotalValence,
        "Mass": rdkit.Chem.rdchem.Atom.GetMass,
        "IsInRing": rdkit.Chem.rdchem.Atom.IsInRing,
        "Hybridization": rdkit.Chem.rdchem.Atom.GetHybridization,
        "ChiralTag": rdkit.Chem.rdchem.Atom.GetChiralTag,
        "FormalCharge": rdkit.Chem.rdchem.Atom.GetFormalCharge,
        "ImplicitValence": rdkit.Chem.rdchem.Atom.GetImplicitValence,
        "NumRadicalElectrons": rdkit.Chem.rdchem.Atom.GetNumRadicalElectrons,
        "Idx": rdkit.Chem.rdchem.Atom.GetIdx,
        "CIPCode": lambda atom: atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else False,
        "ChiralityPossible": lambda atom: atom.HasProp('_ChiralityPossible')
    }

    bond_fun_dict = {
        "BondType": rdkit.Chem.rdchem.Bond.GetBondType,
        "IsAromatic": rdkit.Chem.rdchem.Bond.GetIsAromatic,
        "IsConjugated": rdkit.Chem.rdchem.Bond.GetIsConjugated,
        "IsInRing": rdkit.Chem.rdchem.Bond.IsInRing,
        "Stereo": rdkit.Chem.rdchem.Bond.GetStereo
    }

    mol_fun_dict = {
        "ExactMolWt": rdkit.Chem.Descriptors.ExactMolWt,
        "NumAtoms": lambda mol_arg_lam: mol_arg_lam.GetNumAtoms()
    }

    def __init__(self, mol=None, add_hydrogen: bool = True, is_directed: bool = False):
        """Initialize MolecularGraphRDKit with mol object.

        Args:
            mol (rdkit.Chem.rdchem.Mol): Mol object from rdkit. Default is None.
            add_hydrogen (bool): Whether to add hydrogen. Default is True.
            is_directed (bool): Whether the edges are directed. Default is False.
        """
        super().__init__(mol=mol, add_hydrogen=add_hydrogen)
        self.mol = mol
        self._add_hydrogen = add_hydrogen
        self.is_directed = is_directed

    def from_smiles(self, smile, sanitize: bool = True):
        """Make molecule from smile.

        Args:
            smile (str): Smile string for the molecule.
            sanitize (bool): Whether to sanitize molecule.
        """
        # Make molecule from smile via rdkit
        m = rdkit.Chem.MolFromSmiles(smile)
        if sanitize:
            rdkit.Chem.SanitizeMol(m)
        if self._add_hydrogen:
            m = rdkit.Chem.AddHs(m)  # add H's to the molecule

        m.SetProp("_Name", smile)
        self.mol = m
        return self

    @property
    def node_coordinates(self):
        m = self.mol
        if len(m.GetConformers()) > 0:
            return np.array(self.mol.GetConformers()[0].GetPositions())
        try:
            rdkit.Chem.AllChem.EmbedMolecule(m, useRandomCoords=True)
            rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)
            rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
            rdkit.Chem.AssignStereochemistryFrom3D(m)
            rdkit.Chem.AssignStereochemistry(m)
            self.mol = m
        except ValueError:
            try:
                rdkit.Chem.RemoveStereochemistry(m)
                rdkit.Chem.AssignStereochemistry(m)
                rdkit.Chem.AllChem.EmbedMolecule(m, useRandomCoords=True)
                rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)
                rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
                rdkit.Chem.AssignStereochemistryFrom3D(m)
                rdkit.Chem.AssignStereochemistry(m)
                self.mol = m
            except ValueError:
                print("WARNING:kgcnn: Rdkit could not embed molecule with smile",  m.GetProp("_Name"))
                return []

        return np.array(self.mol.GetConformers()[0].GetPositions())

    def from_mol_block(self, mb, sanitize=False):
        self.mol = rdkit.Chem.MolFromMolBlock(mb, removeHs=(not self._add_hydrogen), sanitize=sanitize)
        return self

    def to_mol_block(self):
        return rdkit.Chem.MolToMolBlock(self.mol)

    @property
    def node_symbol(self):
        return [rdkit.Chem.rdchem.Atom.GetSymbol(x) for x in self.mol.GetAtoms()]

    @property
    def node_number(self):
        return np.array([rdkit.Chem.rdchem.Atom.GetAtomicNum(x) for x in self.mol.GetAtoms()])

    @property
    def edge_indices(self):
        m = self.mol
        bond_idx = []
        for i, x in enumerate(m.GetBonds()):
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
            if not self.is_directed:
                # Add a bond with opposite direction but same properties
                bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
        # Sort directed bonds
        bond_idx = np.array(bond_idx, dtype="int64")
        if len(bond_idx) > 0:
            order1 = np.argsort(bond_idx[:, 1], axis=0, kind='mergesort')  # stable!
            ind1 = bond_idx[order1]
            order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')  # stable!
            ind2 = ind1[order2]
            # Take the sorted bonds
            bond_idx = ind2
        return bond_idx

    def edge_attributes(self, properties: list, encoder: dict):
        m = self.mol
        edges = self._check_properties_list(properties, sorted(self.bond_fun_dict.keys()), "Bond")
        encoder = self._check_encoder(encoder, sorted(self.bond_fun_dict.keys()))

        # Collect info about bonds
        bond_info = []
        bond_idx = []
        for i, x in enumerate(m.GetBonds()):
            attr = []
            for k in edges:
                attr += encoder[k](self.bond_fun_dict[k](x)) if k in encoder else [self.bond_fun_dict[k](x)]
            bond_info.append(attr)
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
            # Add a bond with opposite direction but same properties
            if not self.is_directed:
                bond_info.append(attr)
                bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])

        # Sort directed bonds
        bond_idx = np.array(bond_idx, dtype="int64")
        if len(bond_idx) > 0:
            order1 = np.argsort(bond_idx[:, 1], axis=0, kind='mergesort')  # stable!
            ind1 = bond_idx[order1]
            val1 = [bond_info[i] for i in order1]
            order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')  # stable!
            ind2 = ind1[order2]
            val2 = [val1[i] for i in order2]
            # Take the sorted bonds
            bond_idx = ind2
            bond_info = val2
        return bond_idx, bond_info

    def node_attributes(self, properties: list, encoder: dict):
        m = self.mol
        nodes = self._check_properties_list(properties, sorted(self.atom_fun_dict.keys()), "Atom")
        encoder = self._check_encoder(encoder, sorted(self.atom_fun_dict.keys()))
        # Collect info about atoms
        atom_info = []
        for i, atm in enumerate(m.GetAtoms()):
            attr = []
            for k in nodes:
                attr += encoder[k](self.atom_fun_dict[k](atm)) if k in encoder else [self.atom_fun_dict[k](atm)]
            atom_info.append(attr)
        return atom_info

    def graph_attributes(self, properties: list, encoder: dict):
        graph = self._check_properties_list(properties, sorted(self.mol_fun_dict.keys()), "Molecule")
        m = self.mol
        encoder = self._check_encoder(encoder, sorted(self.mol_fun_dict.keys()))
        # Mol info
        mol_info = []
        for k in graph:
            mol_info += encoder[k](self.mol_fun_dict[k](m)) if k in encoder else [self.mol_fun_dict[k](m)]
        return mol_info
