import numpy as np
import os
import logging
from kgcnn.molecule.base import MolGraphInterface
from openbabel import openbabel

# Module logger.
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


# OpenBabel can not be installed with pip but conda only. It is an optional dependency at the moment.
# There was a problem with OpenBabel not finding the 'BABEL_DATADIR'. One can set it manually with
# python `os.environ['BABEL_DATADIR'] = ...` .
if "BABEL_DATADIR" not in os.environ:
    module_logger.warning(
        "System variable 'BABEL_DATADIR' is not set. Please set `os.environ['BABEL_DATADIR'] = ...` manually.")


class MolecularGraphOpenBabel(MolGraphInterface):
    r"""A graph object representing a strict molecular graph, e.g. only chemical bonds.
    This class is an interface to :obj:`OBMol` class to retrieve graph properties.

    .. code-block:: python

        import numpy as np
        from kgcnn.mol.graph_babel import MolecularGraphOpenBabel
        mg = MolecularGraphOpenBabel()
        mg.from_smiles("CC(C)C(C(=O)O)N")
        mg.add_hs()
        mg.make_conformer()
        mg.optimize_conformer()
        mg.compute_partial_charges()
        print(MolecularGraphOpenBabel.atom_fun_dict.keys(), MolecularGraphOpenBabel.bond_fun_dict.keys())
        print(mg.node_coordinates)
        print(mg.edge_indices)
        print(mg.node_attributes(properties=["NumBonds", "GasteigerCharge"], encoder={}))

    """

    atom_fun_dict = {
        "IsInRing": lambda atom: atom.IsInRing(),
        "IsPeriodic": lambda atom: atom.IsPeriodic(),
        "IsHetAtom": lambda atom: atom.IsHetAtom(),
        "IsAxial": lambda atom: atom.IsAxial(),
        "IsMetal": lambda atom: atom.IsMetal(),
        "IsAmideNitrogen": lambda atom: atom.IsAmideNitrogen(),
        "IsAromatic": lambda atom: atom.IsAromatic(),
        "IsAromaticNOxide": lambda atom: atom.IsAromaticNOxide(),
        "IsCarboxylOxygen": lambda atom: atom.IsCarboxylOxygen(),
        "IsChiral": lambda atom: atom.IsChiral(),
        "IsHbondAcceptor": lambda atom: atom.IsHbondAcceptor(),
        "IsHbondAcceptorSimple": lambda atom: atom.IsHbondAcceptorSimple(),
        "IsHbondDonor": lambda atom: atom.IsHbondDonor(),
        "IsHbondDonorH": lambda atom: atom.IsHbondDonorH(),
        "IsHeteroatom": lambda atom: atom.IsHeteroatom(),
        "IsInRingSize6": lambda atom: atom.IsInRingSize(6),
        "IsInRingSize5": lambda atom: atom.IsInRingSize(5),
        "IsNitroOxygen": lambda atom: atom.IsNitroOxygen(),
        "IsNonPolarHydrogen": lambda atom: atom.IsNonPolarHydrogen(),
        "IsPhosphateOxygen": lambda atom: atom.IsPhosphateOxygen(),
        "IsPolarHydrogen": lambda atom: atom.IsPolarHydrogen(),
        "IsSulfateOxygen": lambda atom: atom.IsSulfateOxygen(),
        "Visit": lambda atom: atom.Visit,
        "Isotope": lambda atom: atom.GetIsotope(),
        "Data": lambda bond: bond.GetData(),
        "X": lambda bond: bond.GetX(),
        "Y": lambda bond: bond.GetY(),
        "Z": lambda bond: bond.GetZ(),
        "ExplicitValence": lambda bond: bond.GetExplicitValence(),
        "TotalDegree": lambda bond: bond.GetTotalDegree(),
        "AtomicMass": lambda bond: bond.GetAtomicMass(),
        "AtomicNum": lambda bond: bond.GetAtomicNum(),
        "Coordinate": lambda bond: bond.GetCoordinate(),
        "CoordinateIdx": lambda bond: bond.GetCoordinateIdx(),
        "ExactMass": lambda bond: bond.GetExactMass(),
        "ExplicitDegree": lambda bond: bond.GetExplicitDegree(),
        "FormalCharge": lambda bond: bond.GetFormalCharge(),
        "HeteroDegree": lambda bond: bond.GetHeteroDegree(),
        "HvyDegree": lambda bond: bond.GetHvyDegree(),
        "Hyb": lambda bond: bond.GetHyb(),
        "ImplicitHCount": lambda bond: bond.GetImplicitHCount(),
        "Index": lambda bond: bond.GetIndex(),
        "PartialCharge": lambda bond: bond.GetPartialCharge(),
        "Residue": lambda bond: bond.GetResidue(),
        "SpinMultiplicity": lambda bond: bond.GetSpinMultiplicity(),
        "Title": lambda bond: bond.GetTitle(),
        "TotalValence": lambda bond: bond.GetTotalValence(),
        "Type": lambda bond: bond.GetType(),
        "Vector": lambda bond: bond.GetVector(),
        "HasResidue": lambda bond: bond.HasResidue(),
        "HasDoubleBond": lambda bond: bond.HasDoubleBond(),
        "HasSingleBond": lambda bond: bond.HasSingleBond(),
        "HasAromaticBond": lambda bond: bond.HasAromaticBond(),
        "HasAlphaBetaUnsat": lambda bond: bond.HasAlphaBetaUnsat(),
        "HasBondOfOrder3": lambda bond: bond.HasBondOfOrder(3),
        "HasBondOfOrder2": lambda bond: bond.HasBondOfOrder(2),
        "HasBondOfOrder1": lambda bond: bond.HasBondOfOrder(1),
        "HasNonSingleBond": lambda bond: bond.HasNonSingleBond()
    }
    bond_fun_dict = {
        "BondOrder": lambda bond: bond.GetBondOrder(),
        "IsAromatic": lambda bond: bond.IsAromatic(),
        "IsInRing": lambda bond: bond.IsInRing(),
        "Idx": lambda bond: bond.GetIdx(),
        "Id": lambda bond: bond.GetId(),
        "BeginAtom": lambda bond: bond.GetBeginAtom(),
        "BeginAtomIdx": lambda bond: bond.GetBeginAtomIdx(),
        "EndAtom": lambda bond: bond.GetEndAtom(),
        "EndAtomIdx": lambda bond: bond.GetEndAtomIdx(),
        "Flags": lambda bond: bond.GetFlags(),
        "Parent": lambda bond: bond.GetParent(),
        "Length": lambda bond: bond.GetLength(),
        "EquibLength": lambda bond: bond.GetEquibLength(),
        "Aromatic": lambda bond: bond.Aromatic,
        "CisOrTrans": lambda bond: bond.CisOrTrans,
        "IsHash": lambda bond: bond.IsHash(),
        "IsAmide": lambda bond: bond.IsAmide(),
        "IsEster": lambda bond: bond.IsEster(),
        "IsCarbonyl": lambda bond: bond.IsCarbonyl(),
        "IsCisOrTrans": lambda bond: bond.IsCisOrTrans(),
        "IsClosure": lambda bond: bond.IsClosure(),
        "IsDoubleBondGeometry": lambda bond: bond.IsDoubleBondGeometry(),
        "IsPeriodic": lambda bond: bond.IsPeriodic(),
        "IsPrimaryAmide": lambda bond: bond.IsPrimaryAmide(),
        "IsTertiaryAmide": lambda bond: bond.IsTertiaryAmide(),
        "IsWedge": lambda bond: bond.IsWedge(),
        "IsWedgeOrHash": lambda bond: bond.IsWedgeOrHash(),
        "Visit": lambda bond: bond.Visit,
    }
    mol_fun_dict = {
        "NumAtoms": lambda m: m.NumAtoms(),
        "NumBonds": lambda m: m.NumBonds(),
        "ExactMass": lambda m: m.GetExactMass(),
        "TotalCharge": lambda m: m.GetTotalCharge(),
    }

    def __init__(self, mol=None, make_directed: bool = False):
        """Set the mol attribute for composition. This mol instances will be the backends molecule class.

        Args:
            mol (openbabel.OBMol): OpenBabel molecule.
            make_directed (bool): Whether the edges are directed. Default is False.
        """
        super().__init__(mol=mol, make_directed=make_directed)
        self.mol = mol

    def make_conformer(self, **kwargs):
        r"""Make conformer for mol-object.

        Args:
            kwargs: Not used.

        Returns:
            bool: Whether conformer generation was successful
        """
        if self.mol is None:
            return False
        builder = openbabel.OBBuilder()
        build_okay = builder.Build(self.mol)
        return build_okay

    def optimize_conformer(self, force_field="mmff94", steps=100, **kwargs):
        r"""Optimize conformer. Requires an initial conformer. See :obj:`make_conformer`.

        Args:
            force_field (str): Force field type.
            steps (int): Number of iteration steps.
            kwargs: Kwargs for SteepestDescent.

        Returns:
            bool: Whether conformer optimization was successful.
        """
        if self.mol is None:
            return False
        ff = openbabel.OBForceField.FindType(force_field)
        ff_setup_okay = ff.Setup(self.mol)
        ff.SteepestDescent(steps, **kwargs)  # defaults are 50-500 in pybel
        ff.GetCoordinates(self.mol)
        return ff_setup_okay

    def add_hs(self, **kwargs):
        """Add Hydrogen."""
        self.mol.AddHydrogens(**kwargs)

    def remove_hs(self, **kwargs):
        """Remove Hydrogen."""
        self.mol.DeleteHydrogens(**kwargs)

    def compute_partial_charges(self, method="gasteiger", **kwargs):
        """Compute partial charges.

        Args:
            method (str): Name of charge model.
            kwargs: Not used.

        Returns:
            bool: Compute charges return value.
        """
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
            xyz_string: String of xyz block.

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
