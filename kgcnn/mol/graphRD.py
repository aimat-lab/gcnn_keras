import numpy as np
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.Descriptors
import logging

# For this module please install rdkit. See: https://www.rdkit.org/docs/Install.html
# or check https://pypi.org/project/rdkit-pypi/ via `pip install rdkit-pypi`
# or try `conda install -c rdkit rdkit` or `conda install -c conda-forge rdkit`
from kgcnn.mol.base import MolGraphInterface

# Module logger
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class MolecularGraphRDKit(MolGraphInterface):
    r"""A graph object representing a strict molecular graph, e.g. only chemical bonds using a mol-object from `RDkit`
    chemical informatics package. Generate attributes for nodes, edges, and graph which are in a molecular graph atoms,
    bonds and the molecule itself. Naming should

    """

    # Dictionary of predefined atom or node properties
    atom_fun_dict = {
        "AtomicNum": lambda atom: atom.GetAtomicNum(),
        "Symbol": lambda atom: atom.GetSymbol(),
        "NumExplicitHs": lambda atom: atom.GetNumExplicitHs(),
        "NumImplicitHs": lambda atom: atom.GetNumImplicitHs(),
        "TotalNumHs": lambda atom: atom.GetTotalNumHs(),
        "IsAromatic": lambda atom: atom.GetIsAromatic(),
        "TotalDegree": lambda atom: atom.GetTotalDegree(),
        "TotalValence": lambda atom: atom.GetTotalValence(),
        "Mass": lambda atom: atom.GetMass(),
        "IsInRing": lambda atom: atom.IsInRing(),
        "Hybridization": lambda atom: atom.GetHybridization(),
        "ChiralTag": lambda atom: atom.GetChiralTag(),
        "FormalCharge": lambda atom: atom.GetFormalCharge(),
        "ImplicitValence": lambda atom: atom.GetImplicitValence(),
        "NumRadicalElectrons": lambda atom: atom.GetNumRadicalElectrons(),
        "Idx": lambda atom: atom.GetIdx(),
        "CIPCode": lambda atom: atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else False,
        "ChiralityPossible": lambda atom: atom.HasProp('_ChiralityPossible')
    }

    # Dictionary of predefined bond or edge properties
    bond_fun_dict = {
        "BondType": lambda bond: bond.GetBondType(),
        "IsAromatic": lambda bond: bond.GetIsAromatic(),
        "IsConjugated": lambda bond: bond.GetIsConjugated(),
        "IsInRing": lambda bond: bond.IsInRing(),
        "Stereo": lambda bond: bond.GetStereo()
    }

    # Dictionary of predefined molecule or graph-level properties
    mol_fun_dict = {
        "ExactMolWt": rdkit.Chem.Descriptors.ExactMolWt,
        "NumAtoms": lambda mol_arg_lam: mol_arg_lam.GetNumAtoms()
    }

    def __init__(self, mol=None, add_hydrogen: bool = True, make_directed: bool = False,
                 make_conformer: bool = True, optimize_conformer: bool = True):
        """Initialize MolecularGraphRDKit with mol object.

        Args:
            mol (rdkit.Chem.rdchem.Mol): Mol object from rdkit. Default is None.
            add_hydrogen (bool): Whether to add hydrogen. Default is True.
            make_directed (bool): Whether the edges are directed. Default is False.
            make_conformer (bool): Whether to make conformers. Default is True.
            optimize_conformer (bool): Whether to optimize the conformer with standard force field.
        """
        super().__init__(mol=mol, add_hydrogen=add_hydrogen)
        self.mol = mol
        self._add_hydrogen = add_hydrogen
        self._make_directed = make_directed
        self._make_conformer = make_conformer
        self._optimize_conformer = optimize_conformer

    def make_conformer(self, useRandomCoords: bool = True):
        """Make conformer for mol-object.

        Args:
            useRandomCoords (bool): Whether ot use random coordinates. Default is True.

        Returns:
            bool: Whether conformer generation was successful
        """
        if self.mol is None:
            return False
        m = self.mol
        try:
            rdkit.Chem.RemoveStereochemistry(m)
            rdkit.Chem.AssignStereochemistry(m)
            rdkit.Chem.AllChem.EmbedMolecule(m, useRandomCoords=useRandomCoords)
            return True
        except ValueError:
            logging.warning("Rdkit could not embed molecule %s" % m.GetProp("_Name"))
            return False

    def optimize_conformer(self):
        r"""Optimize conformer. Requires a initial conformer. See :obj:`make_conformer`.

        Returns:
            bool: Whether conformer generation was successful
        """
        if self.mol is None:
            return False
        m = self.mol
        try:
            rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)
            rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
            rdkit.Chem.AssignStereochemistryFrom3D(m)
            rdkit.Chem.AssignStereochemistry(m)
            return True
        except ValueError:
            module_logger.error("`RDkit` could not optimize conformer %s" % m.GetProp("_Name"))
            return False

    def from_smiles(self, smile, sanitize: bool = True):
        r"""Make molecule from smile. Adding of hydrogen and generating and optimizing a conformer is determined by
        flags in initialization of this class via :obj:`_add_hydrogen` etc.

        Args:
            smile (str): Smile string for the molecule.
            sanitize (bool): Whether to sanitize molecule.

        Returns:
            self
        """
        # Make molecule from smile via rdkit
        m = rdkit.Chem.MolFromSmiles(smile)
        if m is None:
            module_logger.error("Rdkit can not convert smile %s" % smile)
            return self

        if sanitize:
            rdkit.Chem.SanitizeMol(m)
        if self._add_hydrogen:
            m = rdkit.Chem.AddHs(m)  # add H's to the molecule

        m.SetProp("_Name", smile)
        self.mol = m

        if self._make_conformer:
            self.make_conformer()
        if self._optimize_conformer:
            self.optimize_conformer()
        return self

    @property
    def node_coordinates(self):
        """Return a list or array of atomic coordinates of the molecule."""
        m = self.mol
        if len(m.GetConformers()) > 0:
            return np.array(self.mol.GetConformers()[0].GetPositions())
        return []

    def to_smiles(self):
        """Return a smile string representation of the mol instance.

        Returns:
            smile (str): Smile string.
        """
        if self.mol is None:
            return
        return rdkit.Chem.MolToSmiles(self.mol)

    def from_mol_block(self, mol_block, sanitize: bool = False):
        r"""Set mol-instance from a mol-block string. Removing hydrogen is controlled by flags defined by the
        initialization of this instance via :obj:`_add_hydrogen`.

        Args:
            mol_block (str): Mol-block representation of a molecule.
            sanitize (bool): Whether to sanitize the mol-object.

        Returns:
            self
        """
        if mol_block is None or len(mol_block) == 0:
            module_logger.error("Can not make mol-object for mol string %s" % mol_block)
            return self
        self.mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=(not self._add_hydrogen), sanitize=sanitize)
        return self

    def to_mol_block(self):
        """Make mol-block from mol-object.

        Returns:
            mol_block (str): Mol-block representation of a molecule.
        """
        if self.mol is None:
            module_logger.error("Can not make mol string %s" % self.mol)
            return None
        return rdkit.Chem.MolToMolBlock(self.mol)

    @property
    def node_symbol(self):
        """Return a list of atomic symbols of the molecule."""
        return [x.GetSymbol() for x in self.mol.GetAtoms()]

    @property
    def node_number(self):
        """Return list of node number which is the atomic number of each atom in the molecule"""
        return np.array([x.GetAtomicNum() for x in self.mol.GetAtoms()])

    @property
    def edge_number(self):
        """Make list of the bond order or type of each bond in the molecule."""
        temp = self.edge_attributes(["BondType"], encoder={"BondType": int})
        return temp

    @property
    def edge_indices(self):
        r"""Return edge or bond indices of the molecule. If flag :obj:`_make_directed` is set to true, then only the
        bonds as defined by `RDkit` are returned, otherwise a table of sorted undirected bond indices is returned.

        Returns:
            np.ndarray: Array of bond indices.
        """
        m = self.mol
        bond_idx = []
        for i, x in enumerate(m.GetBonds()):
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
            if not self._make_directed:
                # Add a bond with opposite direction but same properties
                bond_idx.append([x.GetBeginAtomIdx(), x.GetEndAtomIdx()])
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
        r"""Return edge or bond attributes together with bond indices of the molecule.
        If flag :obj:`_make_directed` is set to true, then only the bonds as defined by `RDkit` are returned,
        otherwise a table of sorted undirected bond indices is returned.

        Args:
            properties (list): List of identifiers for properties to retrieve from bonds, or
                a callable object that receives `RDkit` bond class and returns list or value.
            encoder (dict): A dictionary of optional encoders for each string identifier.

        Returns:
            tuple: Indices, Attributes.
        """
        m = self.mol
        edges = self._check_properties_list(properties, sorted(self.bond_fun_dict.keys()), "Bond")
        encoder = self._check_encoder(encoder, sorted(self.bond_fun_dict.keys()))

        # Collect info about bonds
        bond_info = []
        bond_idx = []
        for i, x in enumerate(m.GetBonds()):
            attr = []
            for k in edges:
                temp = encoder[k](self.bond_fun_dict[k](x)) if k in encoder else self.bond_fun_dict[k](x)
                if isinstance(temp, (list, tuple)):
                    attr += list(temp)
                else:
                    attr.append(temp)
            bond_info.append(attr)
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
            # Add a bond with opposite direction but same properties
            if not self._make_directed:
                bond_info.append(attr)
                bond_idx.append([x.GetBeginAtomIdx(), x.GetEndAtomIdx()])

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
        r"""Return node or atom attributes.

        Args:
            properties (list): List of string identifiers for properties to retrieve from atoms, or
                a callable object that receives `RDkit` atom class and returns list or value.
            encoder (dict): A dictionary of optional encoders for each string identifier.

        Returns:
            list: List of atomic properties.
        """
        m = self.mol
        nodes = self._check_properties_list(properties, sorted(self.atom_fun_dict.keys()), "Atom")
        encoder = self._check_encoder(encoder, sorted(self.atom_fun_dict.keys()))
        # Collect info about atoms
        atom_info = []
        for i, atm in enumerate(m.GetAtoms()):
            attr = []
            for k in nodes:
                if isinstance(k, str):
                    temp = encoder[k](self.atom_fun_dict[k](atm)) if k in encoder else self.atom_fun_dict[k](atm)
                else:
                    temp = k(atm)
                if isinstance(temp, (list, tuple)):
                    attr += list(temp)
                else:
                    attr.append(temp)
            atom_info.append(attr)
        return atom_info

    def graph_attributes(self, properties: list, encoder: dict):
        r"""Return graph or molecular attributes.

        Args:
            properties (list): List of identifiers for properties to retrieve from the molecule, or
                a callable object that receives `RDkit` molecule class and returns list or value.
            encoder (dict): A dictionary of optional encoders for each string identifier.

        Returns:
            list: List of atomic properties.
        """
        graph = self._check_properties_list(properties, sorted(self.mol_fun_dict.keys()), "Molecule")
        m = self.mol
        encoder = self._check_encoder(encoder, sorted(self.mol_fun_dict.keys()))
        # Mol info
        attr = []
        for k in graph:
            if isinstance(k, str):
                temp = encoder[k](self.mol_fun_dict[k](m)) if k in encoder else self.mol_fun_dict[k](m)
            else:
                temp = k(m)
            if isinstance(temp, (list, tuple)):
                attr += list(temp)
            else:
                attr.append(temp)
        return attr
