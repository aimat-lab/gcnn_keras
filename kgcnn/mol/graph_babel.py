import numpy as np
import os
import logging
from kgcnn.mol.base import MolGraphInterface
from openbabel import openbabel

# Module logger.
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

if "BABEL_DATADIR" not in os.environ:
    module_logger.warning(
        "System variable 'BABEL_DATADIR' is not set. Please set `os.environ['BABEL_DATADIR'] = ...` manually.")


class MolecularGraphOpenBabel(MolGraphInterface):
    r"""A graph object representing a strict molecular graph, e.g. only chemical bonds."""

    atom_fun_dict = {}
    mol_fun_dict = {}
    bond_fun_dict = {}

    def __init__(self, mol=None, make_directed: bool = False):
        """Set the mol attribute for composition. This mol instances will be the backends molecule class.

        Args:
            mol (openbabel.OBMol): OpenBabel molecule.
            make_directed (bool): Whether the edges are directed. Default is False.
        """
        super().__init__(mol=mol, make_directed=make_directed)
        self.mol = mol

    def make_conformer(self, **kwargs):
        if self.mol is None:
            return False
        builder = openbabel.OBBuilder(**kwargs)
        build_okay = builder.Build(self.mol)
        return build_okay

    def optimize_conformer(self):
        if self.mol is None:
            return False
        ff = openbabel.OBForceField.FindType("mmff94")
        ff_setup_okay = ff.Setup(self.mol)
        ff.SteepestDescent(100)  # defaults are 50-500 in pybel
        ff.GetCoordinates(self.mol)
        return ff_setup_okay

    def add_hs(self, **kwargs):
        self.mol.AddHydrogens(**kwargs)

    def remove_hs(self, **kwargs):
        self.mol.DeleteHydrogens(**kwargs)

    def compute_charges(self, method="gasteiger", **kwargs):
        mol = self.mol
        ob_charge_model = openbabel.OBChargeModel.FindType(method)
        return ob_charge_model.ComputeCharges(mol)

    def from_smiles(self, smile: str, sanitize: bool = True):
        """Make molecule from smile.

        Args:
            smile (str): Smile string for the molecule.
            sanitize (bool): Whether to sanitize molecule.
        """
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat("smiles")
        self.mol = openbabel.OBMol()
        ob_conversion.ReadString(self.mol, smile)
        return self

    def to_smiles(self):
        """Return a smile string representation of the mol instance.

        Returns:
            smile (str): Smile string.
        """
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetOutFormat("smiles")
        return ob_conversion.WriteString(self.mol)

    def from_mol_block(self, mol_block: str, keep_hs: bool = True, sanitize: bool = True):
        """Set mol-instance from a string representation containing coordinates and bond information that is MDL mol
        format equivalent.

        Args:
            mol_block (str): Mol-block representation of a molecule.
            sanitize (bool): Whether to sanitize the mol-object.
            keep_hs (bool): Whether to keep hydrogen.

        Returns:
            self
        """
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat("mol")
        self.mol = openbabel.OBMol()
        ob_conversion.ReadString(self.mol, mol_block)
        if self.mol.HasHydrogensAdded() and not keep_hs:
            self.mol.DeleteHydrogens()
        return self

    def from_xyz(self, xyz_string):
        """Setting mol-instance from an external xyz-string. Does not add hydrogen or makes conformers.

        Args:
            xyz_string:

        Returns:
            self
        """
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat("xyz")
        self.mol = openbabel.OBMol()
        ob_conversion.ReadString(self.mol, xyz_string)
        return self

    def to_mol_block(self):
        """Make a more extensive string representation containing coordinates and bond information from self.

        Returns:
            mol_block (str): Mol-block representation of a molecule.
        """
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetOutFormat("mol")
        return ob_conversion.WriteString(self.mol)

    @property
    def node_number(self):
        """Return list of node numbers which is the atomic number of atoms in the molecule"""
        atom_num = []
        for i in range(self.mol.NumAtoms()):
            ats = self.mol.GetAtomById(i)
            # ats = mol.GetAtom(i+1)
            atom_num.append(ats.GetAtomicNum())
        return atom_num

    @property
    def node_symbol(self):
        """Return a list of atomic symbols of the molecule."""
        atom_type = []
        for i in range(self.mol.NumAtoms()):
            ats = self.mol.GetAtomById(i)
            # ats = mol.GetAtom(i+1)
            atom_type.append(ats.GetType())
        return atom_type

    @property
    def node_coordinates(self):
        """Return a list of atomic coordinates of the molecule."""
        xyz = []
        for i in range(self.mol.NumAtoms()):
            ats = self.mol.GetAtomById(i)
            # ats = mol.GetAtom(i+1)
            xyz.append([ats.GetX(), ats.GetY(), ats.GetZ()])
        if len(xyz) <= 0:
            return
        return np.array(xyz)

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

    @property
    def edge_indices(self):
        """Return a list of edge indices of the molecule."""
        bond_idx = []
        for i in range(self.mol.NumBonds()):
            bnd = self.mol.GetBondById(i)
            # bnd = mol.GetBond(i)
            if bnd is None:
                continue
            bond_idx.append([bnd.GetBeginAtomIdx() - 1, bnd.GetEndAtomIdx() - 1])
            if not self._make_directed:
                # Add a bond with opposite direction but same properties
                bond_idx.append([bnd.GetEndAtomIdx() - 1, bnd.GetBeginAtomIdx() - 1])
        # Sort bond indices
        bond_idx, _ = self._sort_bonds(bond_idx)
        return bond_idx

    @property
    def edge_number(self):
        """Return a list of edge number that represents the bond order."""
        bond_number = []
        bond_idx = []
        for i in range(self.mol.NumBonds()):
            bnd = self.mol.GetBondById(i)
            # bnd = mol.GetBond(i)
            if bnd is None:
                continue
            bond_idx.append([bnd.GetBeginAtomIdx() - 1, bnd.GetEndAtomIdx() - 1])
            bond_number.append(bnd.GetBondOrder())
            if not self._make_directed:
                # Add a bond with opposite direction but same properties
                bond_idx.append([bnd.GetEndAtomIdx() - 1, bnd.GetBeginAtomIdx() - 1])
                bond_number.append(bnd.GetBondOrder())
                # Sort bond indices
        bond_idx, bond_number = self._sort_bonds(bond_idx, bond_number)
        return bond_idx, bond_number

    def edge_attributes(self, properties: list, encoder: dict):
        """Make edge attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
        """
        m = self.mol
        edges = self._check_properties_list(properties, sorted(self.bond_fun_dict.keys()), "Bond")
        encoder = self._check_encoder(encoder, sorted(self.bond_fun_dict.keys()))

        # Collect info about bonds
        bond_info = []
        bond_idx = []
        for i in range(m.NumBonds()):
            x = m.GetBondById(i)
            attr = []
            for k in edges:
                if isinstance(k, str):
                    temp = encoder[k](self.bond_fun_dict[k](x)) if k in encoder else self.bond_fun_dict[k](x)
                else:
                    temp = k(x)
                if isinstance(temp, np.ndarray):
                    temp = temp.tolist()
                if isinstance(temp, (list, tuple)):
                    attr += list(temp)
                else:
                    attr.append(temp)
            bond_info.append(attr)
            bond_idx.append([x.GetBeginAtomIdx() - 1, x.GetEndAtomIdx() - 1])
            # Add a bond with opposite direction but same properties
            if not self._make_directed:
                bond_info.append(attr)
                bond_idx.append([x.GetEndAtomIdx() - 1, x.GetBeginAtomIdx() - 1])

        # Sort directed bonds
        bond_idx, bond_info = self._sort_bonds(bond_idx, bond_info)
        return bond_idx, bond_info

    def node_attributes(self, properties: list, encoder: dict):
        """Make node attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
        """
        m = self.mol
        nodes = self._check_properties_list(properties, sorted(self.atom_fun_dict.keys()), "Atom")
        encoder = self._check_encoder(encoder, sorted(self.atom_fun_dict.keys()))
        # Collect info about atoms
        atom_info = []
        for i in range(m.NumAtoms()):
            atm = m.GetAtomById(i)
            attr = []
            for k in nodes:
                if isinstance(k, str):
                    temp = encoder[k](self.atom_fun_dict[k](atm)) if k in encoder else self.atom_fun_dict[k](atm)
                else:
                    temp = k(atm)
                if isinstance(temp, np.ndarray):
                    temp = temp.tolist()
                if isinstance(temp, (list, tuple)):
                    attr += list(temp)
                else:
                    attr.append(temp)
            atom_info.append(attr)
        return atom_info

    def graph_attributes(self, properties: list, encoder: dict):
        """Make graph attributes.

        Args:
            properties (list): List of string identifier for a molecular property. Must match backend features.
            encoder (dict): A dictionary of callable encoder function or class for each string identifier.

        Returns:
            list: List of attributes after processed by the encoder.
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
            if isinstance(temp, np.ndarray):
                temp = temp.tolist()
            if isinstance(temp, (list, tuple)):
                attr += list(temp)
            else:
                attr.append(temp)
        return attr
