import numpy as np
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.Descriptors
import rdkit.Chem.Fragments
import rdkit.Chem.rdMolDescriptors
import rdkit.Chem.rdDetermineBonds
import rdkit.RDLogger
import rdkit.Chem.rdchem
import logging
from typing import Union

# For this module please install rdkit. See: https://www.rdkit.org/docs/Install.html
# or check https://pypi.org/project/rdkit-pypi/ via `pip install rdkit-pypi`
# or try `conda install -c rdkit rdkit` or `conda install -c conda-forge rdkit`
# Note: RDkit version >= 2022.9.2 can be installed with `pip install rdkit` only, successor of `rdkit-pypi`.
# Has been added as requirements to kgcnn.
from kgcnn.molecule.base import MolGraphInterface

# Module logger
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


# Adjust RDLogger
# We may have to intercept warnings from RDkit that are not critical.
# rdkit.RDLogger.DisableLog("rdApp.warning")


class MolecularGraphRDKit(MolGraphInterface):
    r"""A graph object representing a strict molecular graph, e.g. only chemical bonds using a mol-object from
    :obj:`RDkit` chemical informatics package.

    Generate attributes for nodes, edges, and graph which are in a molecular graph atoms, bonds and the molecule itself.
    The class is used to get a graph from a :obj:`RDkit` molecule object but also offers some functionality defined
    in :obj:`MolGraphInterface` .

    .. code-block:: python

        import numpy as np
        from kgcnn.mol.graph_rdkit import MolecularGraphRDKit
        mg = MolecularGraphRDKit()
        mg.from_smiles("CC(C)C(C(=O)O)N")
        mg.add_hs()
        mg.make_conformer()
        mg.optimize_conformer()
        mg.compute_partial_charges()
        print(MolecularGraphRDKit.atom_fun_dict.keys(), MolecularGraphRDKit.bond_fun_dict.keys())
        print(mg.node_coordinates)
        print(mg.edge_indices)
        print(mg.node_attributes(properties=["NumBonds", "GasteigerCharge"], encoder={}))

    """
    _bond_types_name = {
        'AROMATIC': rdkit.Chem.rdchem.BondType.AROMATIC, 'DATIVE': rdkit.Chem.rdchem.BondType.DATIVE,
        'DATIVEL': rdkit.Chem.rdchem.BondType.DATIVEL, 'DATIVEONE': rdkit.Chem.rdchem.BondType.DATIVEONE,
        'DATIVER': rdkit.Chem.rdchem.BondType.DATIVER, 'DOUBLE': rdkit.Chem.rdchem.BondType.DOUBLE,
        'FIVEANDAHALF': rdkit.Chem.rdchem.BondType.FIVEANDAHALF,
        'FOURANDAHALF': rdkit.Chem.rdchem.BondType.FOURANDAHALF, 'HEXTUPLE': rdkit.Chem.rdchem.BondType.HEXTUPLE,
        'HYDROGEN': rdkit.Chem.rdchem.BondType.HYDROGEN, 'IONIC': rdkit.Chem.rdchem.BondType.IONIC,
        'ONEANDAHALF': rdkit.Chem.rdchem.BondType.ONEANDAHALF, 'OTHER': rdkit.Chem.rdchem.BondType.OTHER,
        'QUADRUPLE': rdkit.Chem.rdchem.BondType.QUADRUPLE, 'QUINTUPLE': rdkit.Chem.rdchem.BondType.QUINTUPLE,
        'SINGLE': rdkit.Chem.rdchem.BondType.SINGLE, 'THREEANDAHALF': rdkit.Chem.rdchem.BondType.THREEANDAHALF,
        'THREECENTER': rdkit.Chem.rdchem.BondType.THREECENTER, 'TRIPLE': rdkit.Chem.rdchem.BondType.TRIPLE,
        'TWOANDAHALF': rdkit.Chem.rdchem.BondType.TWOANDAHALF, 'UNSPECIFIED': rdkit.Chem.rdchem.BondType.UNSPECIFIED,
        'ZERO': rdkit.Chem.rdchem.BondType.ZERO}
    _bond_types_values = {
        0: rdkit.Chem.rdchem.BondType.UNSPECIFIED, 1: rdkit.Chem.rdchem.BondType.SINGLE,
        2: rdkit.Chem.rdchem.BondType.DOUBLE, 3: rdkit.Chem.rdchem.BondType.TRIPLE,
        4: rdkit.Chem.rdchem.BondType.QUADRUPLE, 5: rdkit.Chem.rdchem.BondType.QUINTUPLE,
        6: rdkit.Chem.rdchem.BondType.HEXTUPLE, 7: rdkit.Chem.rdchem.BondType.ONEANDAHALF,
        8: rdkit.Chem.rdchem.BondType.TWOANDAHALF, 9: rdkit.Chem.rdchem.BondType.THREEANDAHALF,
        10: rdkit.Chem.rdchem.BondType.FOURANDAHALF, 11: rdkit.Chem.rdchem.BondType.FIVEANDAHALF,
        12: rdkit.Chem.rdchem.BondType.AROMATIC, 13: rdkit.Chem.rdchem.BondType.IONIC,
        14: rdkit.Chem.rdchem.BondType.HYDROGEN, 15: rdkit.Chem.rdchem.BondType.THREECENTER,
        16: rdkit.Chem.rdchem.BondType.DATIVEONE, 17: rdkit.Chem.rdchem.BondType.DATIVE,
        18: rdkit.Chem.rdchem.BondType.DATIVEL, 19: rdkit.Chem.rdchem.BondType.DATIVER,
        20: rdkit.Chem.rdchem.BondType.OTHER, 21: rdkit.Chem.rdchem.BondType.ZERO}

    # Dictionary of predefined atom or node properties
    atom_fun_dict = {
        "NumBonds": lambda atom: len(atom.GetBonds()),
        "AtomicNum": lambda atom: atom.GetAtomicNum(),
        "AtomMapNum": lambda atom: atom.GetAtomMapNum(),
        "Idx": lambda atom: atom.GetIdx(),
        "Degree": lambda atom: atom.GetDegree(),
        "TotalDegree": lambda atom: atom.GetTotalDegree(),
        "Symbol": lambda atom: atom.GetSymbol(),
        "NumExplicitHs": lambda atom: atom.GetNumExplicitHs(),
        "NumImplicitHs": lambda atom: atom.GetNumImplicitHs(),
        "TotalNumHs": lambda atom: atom.GetTotalNumHs(),
        "IsAromatic": lambda atom: atom.GetIsAromatic(),
        "Isotope": lambda atom: atom.GetIsotope(),
        "TotalValence": lambda atom: atom.GetTotalValence(),
        "Mass": lambda atom: atom.GetMass(),
        "MassScaled": lambda atom: float((atom.GetMass() - 10.812) / 116.092),
        "IsInRing": lambda atom: atom.IsInRing(),
        "Hybridization": lambda atom: atom.GetHybridization(),
        "NoImplicit": lambda atom: atom.GetNoImplicit(),
        "ChiralTag": lambda atom: atom.GetChiralTag(),
        "FormalCharge": lambda atom: atom.GetFormalCharge(),
        "ExplicitValence": lambda atom: atom.GetExplicitValence(),
        "ImplicitValence": lambda atom: atom.GetImplicitValence(),
        "NumRadicalElectrons": lambda atom: atom.GetNumRadicalElectrons(),
        "HasOwningMol": lambda atom: atom.HasOwningMol(),
        "PDBResidueInfo": lambda atom: atom.GetPDBResidueInfo(),
        "MonomerInfo": lambda atom: atom.GetMonomerInfo(),
        "Smarts": lambda atom: atom.GetSmarts(),
        "CIPCode": lambda atom: atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None,
        "CIPRank": lambda atom: atom.GetProp('_CIPRank') if atom.HasProp('_CIPRank') else None,
        "ChiralityPossible": lambda atom: atom.GetProp('_ChiralityPossible') if atom.HasProp(
            '_ChiralityPossible') else None,
        "MolFileRLabel": lambda atom: atom.GetProp('_MolFileRLabel') if atom.HasProp('_MolFileRLabel') else None,
        "GasteigerCharge": lambda atom: atom.GetProp('_GasteigerCharge') if atom.HasProp(
            '_GasteigerCharge') else None,
        "GasteigerHCharge": lambda atom: atom.GetProp('_GasteigerHCharge') if atom.HasProp(
            '_GasteigerHCharge') else None,
        "AtomFeatures": lambda atom: rdkit.Chem.rdMolDescriptors.GetAtomFeatures(atom.GetOwningMol(), atom.GetIdx()),
        "DescribeQuery": lambda atom: atom.DescribeQuery(),
        "Rvdw": lambda atom: float(rdkit.Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())),
        "RvdwScaled": lambda atom: float((rdkit.Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6),
        "Rcovalent": lambda atom: float(rdkit.Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())),
        "RcovalentScaled": lambda atom: float(
            (rdkit.Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76),
    }

    # Dictionary of predefined bond or edge properties
    bond_fun_dict = {
        "BondType": lambda bond: bond.GetBondType(),
        "IsAromatic": lambda bond: bond.GetIsAromatic(),
        "IsConjugated": lambda bond: bond.GetIsConjugated(),
        "IsInRing": lambda bond: bond.IsInRing(),
        "Stereo": lambda bond: bond.GetStereo(),
        "Idx": lambda bond: bond.GetIdx(),
        "BeginAtom": lambda bond: bond.GetBeginAtom(),
        "BeginAtomIdx": lambda bond: bond.GetBeginAtomIdx(),
        "BondDir": lambda bond: bond.GetBondDir(),
        "BondTypeAsDouble": lambda bond: bond.GetBondTypeAsDouble(),
        "EndAtom": lambda bond: bond.GetEndAtom(),
        "EndAtomIdx": lambda bond: bond.GetEndAtomIdx(),
        "Smarts": lambda bond: bond.GetSmarts(),
        "DescribeQuery": lambda bond: bond.DescribeQuery(),
    }

    # Dictionary of predefined molecule or graph-level properties and features.
    mol_fun_dict = {
        # Atom counts.
        "C": lambda m: sum([x.GetSymbol() == "C" for x in m.GetAtoms()]),
        "N": lambda m: sum([x.GetSymbol() == "N" for x in m.GetAtoms()]),
        "O": lambda m: sum([x.GetSymbol() == "O" for x in m.GetAtoms()]),
        "H": lambda m: sum([x.GetSymbol() == "H" for x in m.GetAtoms()]),
        "S": lambda m: sum([x.GetSymbol() == "S" for x in m.GetAtoms()]),
        "F": lambda m: sum([x.GetSymbol() == "F" for x in m.GetAtoms()]),
        "Cl": lambda m: sum([x.GetSymbol() == "Cl" for x in m.GetAtoms()]),
        # Counts general.
        "NumAtoms": lambda m: sum([True for _ in m.GetAtoms()]),
        "AtomsIsInRing": lambda m: sum([x.IsInRing() for x in m.GetAtoms()]),
        "AtomsIsAromatic": lambda m: sum([x.GetIsAromatic() for x in m.GetAtoms()]),
        "NumBonds": lambda m: sum([True for _ in m.GetBonds()]),
        "BondsIsConjugated": lambda m: sum([x.GetIsConjugated() for x in m.GetBonds()]),
        "BondsIsAromatic": lambda m: sum([x.GetIsAromatic() for x in m.GetBonds()]),
        "NumRotatableBonds": lambda m: rdkit.Chem.Lipinski.NumRotatableBonds(m),
        # Descriptors
        "ExactMolWt": rdkit.Chem.Descriptors.ExactMolWt,
        "FpDensityMorgan3": lambda m: rdkit.Chem.Descriptors.FpDensityMorgan3(m),
        "FractionCSP3": lambda m: rdkit.Chem.Lipinski.FractionCSP3(m),
        "MolLogP": lambda m: rdkit.Chem.Crippen.MolLogP(m),
        "MolMR": lambda m: rdkit.Chem.Crippen.MolMR(m),
        # Fragments
        "fr_Al_COO": lambda m: rdkit.Chem.Fragments.fr_Al_COO(m),
        "fr_Ar_COO": lambda m: rdkit.Chem.Fragments.fr_Ar_COO(m),
        "fr_Al_OH": lambda m: rdkit.Chem.Fragments.fr_Al_OH(m),
        "fr_Ar_OH": lambda m: rdkit.Chem.Fragments.fr_Ar_OH(m),
        "fr_C_O_noCOO": lambda m: rdkit.Chem.Fragments.fr_C_O_noCOO(m),
        "fr_NH2": lambda m: rdkit.Chem.Fragments.fr_NH2(m),
        "fr_SH": lambda m: rdkit.Chem.Fragments.fr_SH(m),
        "fr_sulfide": lambda m: rdkit.Chem.Fragments.fr_sulfide(m),
        "fr_alkyl_halide": lambda m: rdkit.Chem.Fragments.fr_alkyl_halide(m),
    }

    def __init__(self, mol=None, make_directed: bool = False):
        r"""Initialize :obj:`MolecularGraphRDKit` with mol object.

        Args:
            mol (rdkit.Chem.rdchem.Mol): Mol object from rdkit. Default is None.
            make_directed (bool): Whether the edges are directed. Default is False.

        """
        super().__init__(mol=mol, make_directed=make_directed)

    def make_conformer(self, **kwargs):
        r"""Make conformer for mol-object.

        Args:
            kwargs: Kwargs for rdkit :obj:`EmbedMolecule` .

        Returns:
            bool: Whether conformer generation was successful
        """
        if self.mol is None:
            return False
        m = self.mol
        try:
            rdkit.Chem.AllChem.EmbedMolecule(m, **kwargs)
            return True
        except ValueError:
            logging.warning("`RDkit` could not embed molecule '%s'." % m.GetProp("_Name"))
            return False

    def optimize_conformer(self, force_field="mmff94", **kwargs):
        r"""Optimize conformer. Requires an initial conformer. See :obj:`make_conformer`.

        Args:
            force_field (str): Force field type.
            kwargs: Kwargs for rdkit optimization function. Includes iterations and force field sub type.

        Returns:
            bool: Whether conformer optimization was successful.
        """
        if self.mol is None:
            return False
        m = self.mol

        if force_field[:4] == "mmff":
            try:
                rdkit.Chem.AllChem.MMFFOptimizeMolecule(m, **kwargs)
            except Exception as e:
                module_logger.error("`RDkit` could not optimize conformer '%s' with '%s'." % m.GetProp("_Name", e))
                return False
        elif force_field[:3] == "uff":
            try:
                rdkit.Chem.AllChem.UFFOptimizeMolecule(m, **kwargs)
            except Exception as e:
                module_logger.error("`RDkit` could not optimize conformer '%s' with '%s'." % m.GetProp("_Name", e))
                return False
        else:
            raise ValueError("Unsupported force field '%s'." % force_field)
        return True

    def add_hs(self, **kwargs):
        """Add hydrogen atoms.

        Args:
            kwargs: Kwargs for rdkit method, e.g. can specify explicit or implicit.

        Returns:
            self.
        """
        if self.mol is None:
            module_logger.error("Mol reference is `None`. Can not `add_hs`.")
            return self
        self.mol = rdkit.Chem.AddHs(self.mol, **kwargs)  # add H's to the molecule
        return self

    def remove_hs(self, **kwargs):
        """Remove hydrogen atoms.

        Args:
            kwargs: Kwargs for rdkit method, e.g. can specify explicit or implicit.

        Returns:
            self.
        """
        if self.mol is None:
            module_logger.error("Mol reference is `None`. Can not `remove_hs`.")
            return self
        self.mol = rdkit.Chem.RemoveHs(self.mol, **kwargs)  # add H's to the molecule
        return self

    def clean(self, **kwargs):
        """Clean or sanitize mol."""
        rdkit.Chem.SanitizeMol(self.mol, **kwargs)

    def compute_partial_charges(self, method="gasteiger", **kwargs):
        """Compute partial charges.

        Args:
            method (str): Method to compute partial charges. Defaults to 'gasteiger'.
            **kwargs:

        Returns:
            self
        """
        if self.mol is None:
            module_logger.error("Mol reference is `None`. Can not `compute_charges`.")
            return self
        if method != "gasteiger":
            module_logger.error("Only 'gasteiger' for `compute_charges` is supported.")
        rdkit.Chem.AllChem.ComputeGasteigerCharges(self.mol, **kwargs)
        return self

    def from_smiles(self, smile, sanitize: bool = True, **kwargs):
        r"""Make molecule from smile.

        Args:
            smile (str): Smile string for the molecule.
            sanitize (bool): Whether to sanitize molecule.
            kwargs: Kwargs for :obj:`MolFromSmiles` .

        Returns:
            self
        """
        # Make molecule from smile via rdkit
        m = rdkit.Chem.MolFromSmiles(smile, sanitize=sanitize, **kwargs)
        if m is None:
            module_logger.error("Rdkit can not convert smile %s" % smile)
            return self

        m.SetProp("_Name", smile)
        self.mol = m

        return self

    # noinspection PyPep8Naming
    def from_mol_block(self, mol_block, sanitize: bool = True, keep_hs: bool = True, strictParsing: bool = True):
        r"""Set mol-instance from a mol-block string.

        Args:
            mol_block (str): Mol-block representation of a molecule.
            sanitize (bool): Whether to sanitize the mol-object.
            keep_hs (bool): Whether to keep hydrogen.
            strictParsing (bool): If this is false, the parser is more lax about. correctness of the content.
                Defaults to true.

        Returns:
            self
        """
        if mol_block is None or len(mol_block) == 0:
            module_logger.error("Can not make mol-object for mol string '%s'." % mol_block)
            return self
        self.mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=(not keep_hs), sanitize=sanitize,
                                              strictParsing=strictParsing)

        return self

    def from_xyz(self, xyz_string: str, charge: Union[list, int, None] = None):
        """Setting mol-instance from an external xyz-string. Does not add hydrogen or makes conformers.

        Args:
            xyz_string (str): String of xyz block.
            charge (int, list): Charge or possible charges of the molecule. Default is [0, 1, -1, 2, -2].

        Returns:
            self.
        """
        if xyz_string is None or len(xyz_string) == 0:
            module_logger.error("Can not make mol-object for xyz string '%s'." % xyz_string)
            return self
        if charge is None:
            charge = [0, 1, -1, 2, -2]
        if isinstance(charge, int):
            charge = [charge]
        out_mol = None
        for c in charge:
            try:
                raw_mol = rdkit.Chem.MolFromXYZBlock(xyz_string)
                out_mol = rdkit.Chem.Mol(raw_mol)
                # Can do this in a single call of determine Bonds.
                # rdkit.Chem.rdDetermineBonds.DetermineConnectivity(out_mol, charge=charge)
                rdkit.Chem.rdDetermineBonds.DetermineBonds(out_mol, charge=c)
                break
            except:
                out_mol = None
                continue
        self.mol = out_mol
        return self

    def from_list(self, atoms: Union[list, np.ndarray], bond_idx: Union[list, np.ndarray],
                   bond_order: Union[list, np.ndarray], conformer: Union[list, np.ndarray] = None):
        """

        Args:
            atoms:
            bond_idx:
            bond_order:
            conformer:

        Returns:
            self.
        """
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        m = rdkit.Chem.RWMol()
        for a in atoms:
            m.AddAtom(rdkit.Chem.rdchem.Atom(a))
        # We need undirected bond table.
        bnd_idx, pos = np.unique(np.sort(np.array(bond_idx), axis=-1), axis=0, return_index=True)
        bnd_type = np.array(bond_order)[pos]
        # Add bonds.
        for i, b in zip(bnd_idx, bnd_type):
            m.AddBond(int(i[0]), int(i[1]), self._bond_types_values[int(b)])
        if conformer is not None:
            conf = rdkit.Chem.Conformer(m.GetNumAtoms())
            for i in range(m.GetNumAtoms()):
                conf.SetAtomPosition(i, (conformer[i][0], conformer[i][1], conformer[i][2]))
            m.AddConformer(conf)
        rdkit.Chem.SanitizeMol(m)
        if conformer is not None:
            rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
            rdkit.Chem.AssignStereochemistryFrom3D(m)
        else:
            rdkit.Chem.AssignStereochemistry(m)
        self.mol = m
        return self

    def to_smiles(self, **kwargs):
        """Return a smile string representation of the mol instance.

        Returns:
            smile (str): Smile string.
        """
        if self.mol is None:
            return
        return rdkit.Chem.MolToSmiles(self.mol, **kwargs)

    def to_mol_block(self, **kwargs):
        """Make mol-block from mol-object.

        Returns:
            mol_block (str): Mol-block representation of a molecule.
        """
        if self.mol is None:
            module_logger.error("Can not make mol string %s" % self.mol)
            return None
        return rdkit.Chem.MolToMolBlock(self.mol, **kwargs)

    @property
    def node_coordinates(self):
        """Return a list or array of atomic coordinates of the molecule."""
        m = self.mol
        if len(m.GetConformers()) > 0:
            return np.array(self.mol.GetConformers()[0].GetPositions())
        return None

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
        m = self.mol
        bond_idx = []
        bond_info = []
        for i, x in enumerate(m.GetBonds()):
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
            bond_info.append(int(x.GetBondType()))
            if not self._make_directed:
                # Add a bond with opposite direction but same properties
                bond_idx.append([x.GetBeginAtomIdx(), x.GetEndAtomIdx()])
                bond_info.append(int(x.GetBondType()))
        # Sort directed bonds
        bond_idx, bond_info = self._sort_bonds(bond_idx, bond_info)
        return bond_idx, bond_info

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
        bond_idx, _ = self._sort_bonds(bond_idx)
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
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
            # Add a bond with opposite direction but same properties
            if not self._make_directed:
                bond_info.append(attr)
                bond_idx.append([x.GetBeginAtomIdx(), x.GetEndAtomIdx()])

        # Sort directed bonds
        bond_idx, bond_info = self._sort_bonds(bond_idx, bond_info)
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
                if isinstance(temp, np.ndarray):
                    temp = temp.tolist()
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
            list: List of molecular graph-level properties.
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
