import numpy as np
import os
import logging
from openbabel import openbabel
from kgcnn.mol.base import MolGraphInterface

# Module logger
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

if "BABEL_DATADIR" not in os.environ:
    module_logger.error(
        "System variable 'BABEL_DATADIR' is not set. Please set `os.environ['BABEL_DATADIR'] = ...` manually.")


class MolecularGraphOpenBabel(MolGraphInterface):
    r"""A graph object representing a strict molecular graph, e.g. only chemical bonds. """

    def __init__(self, mol=None, make_directed: bool = False):
        """Set the mol attribute for composition. This mol instances will be the backends molecule class.

        Args:
            mol (openbabel.OBMol): OpenBabel molecule.
            make_directed (bool): Whether the edges are directed. Default is False.
        """
        super().__init__(mol=mol, make_directed=make_directed)
        self.mol = mol

    def make_conformer(self):
        if self.mol is None:
            return False
        builder = openbabel.OBBuilder()
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

    def add_hs(self):
        self.mol.AddHydrogens()

    def remove_hs(self):
        self.mol.DeleteHydrogens()

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
        if self.mol.Has3D():
            xyz = []
            for i in range(self.mol.NumAtoms()):
                ats = self.mol.GetAtomById(i)
                # ats = mol.GetAtom(i+1)
                xyz.append([ats.GetX(), ats.GetY(), ats.GetZ()])
            return np.array(xyz)
        else:
            return None

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
        bond_idx = np.array(bond_idx, dtype="int64")
        if len(bond_idx) > 0:
            order1 = np.argsort(bond_idx[:, 1], axis=0, kind='mergesort')  # stable!
            ind1 = bond_idx[order1]
            order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')  # stable!
            ind2 = ind1[order2]
            # Take the sorted bonds
            bond_idx = ind2
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
        bond_idx = np.array(bond_idx, dtype="int64")
        bond_number = np.array(bond_number, dtype="int64")
        if len(bond_idx) > 0:
            order1 = np.argsort(bond_idx[:, 1], axis=0, kind='mergesort')  # stable!
            ind1 = bond_idx[order1]
            bond_number = bond_number[order1]
            order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')  # stable!
            bond_idx = ind1[order2]
            bond_number = bond_number[order2]
        return bond_idx, bond_number

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


def convert_smile_to_mol_openbabel(smile: str, sanitize: bool = True, add_hydrogen: bool = True,
                                   make_conformers: bool = True, optimize_conformer: bool = True,
                                   stop_logging: bool = False):

    if stop_logging:
        openbabel.obErrorLog.StopLogging()

    try:
        m = openbabel.OBMol()
        ob_conversion = openbabel.OBConversion()
        format_okay = ob_conversion.SetInAndOutFormats("smi", "mol")
        read_okay = ob_conversion.ReadString(m, smile)
        is_okay = {"format_okay": format_okay, "read_okay": read_okay}
        if make_conformers:
            # We need to make conformer with builder
            builder = openbabel.OBBuilder()
            build_okay = builder.Build(m)
            is_okay.update({"build_okay": build_okay})
        if add_hydrogen:
            # it seems h's are made after build, an get embedded too
            m.AddHydrogens()
        if optimize_conformer and make_conformers:
            ff = openbabel.OBForceField.FindType("mmff94")
            ff_setup_okay = ff.Setup(m)
            ff.SteepestDescent(100)  # defaults are 50-500 in pybel
            ff.GetCoordinates(m)
            is_okay.update({"ff_setup_okay": ff_setup_okay})
        all_okay = all(list(is_okay.values()))
        if not all_okay:
            print("WARNING: Openbabel returned false flag %s" % [key for key, value in is_okay.items() if not value])
    except:
        m = None
        ob_conversion = None

    # Set back to default
    if stop_logging:
        openbabel.obErrorLog.StartLogging()

    if m is not None:
        return ob_conversion.WriteString(m)
    return None


def convert_xyz_to_mol_openbabel(xyz_string: str, stop_logging: bool = False):
    """Convert xyz-string to mol-string.

    The order of atoms in the list should be the same as output. Uses openbabel for conversion.

    Args:
        xyz_string (str): Convert the xyz string to mol-string
        stop_logging (bool): Whether to stop logging. Default is False.

    Returns:
        str: Mol-string. Generates bond information in addition to coordinates from xyz-string.
    """
    if stop_logging:
        openbabel.obErrorLog.StopLogging()

    ob_conversion = openbabel.OBConversion()
    ob_conversion.SetInAndOutFormats("xyz", "mol")
    # ob_conversion.SetInFormat("xyz")

    mol = openbabel.OBMol()
    ob_conversion.ReadString(mol, xyz_string)
    # print(xyz_str)

    out_mol = ob_conversion.WriteString(mol)

    # Set back to default
    if stop_logging:
        openbabel.obErrorLog.StartLogging()
    return out_mol
