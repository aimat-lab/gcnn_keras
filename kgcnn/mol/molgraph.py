import numpy as np
try:
    import rdkit
    import rdkit.Chem
    import rdkit.Chem.AllChem
    import rdkit.Chem.Descriptors
except ImportError:
    print("ERROR:kgcnn: This module needs Rdkit To be installed")

# For this module please install rdkit. See: https://www.rdkit.org/docs/Install.html
# or check https://pypi.org/project/rdkit-pypi/ via `pip install rdkit-pypi`
# or try `conda install -c rdkit rdkit` or `conda install -c conda-forge rdkit`
from kgcnn.mol.base import MolGraphInterface


class OneHotEncoder:
    """Simple One-Hot-Encoding for python lists. Uses a list of possible values for a one-hot encoding of a single
    value. The translated values must support ``__eq__()`` operator. The list of possible values must be set beforehand.
    Is used as a basic encoder example for ``MolecularGraphRDKit``. There can not be different dtypes in categories.
    """
    _dtype_translate = {"int": int, "float": float, "str": str}

    def __init__(self, categories: list, add_unknown: bool = True, dtype: str = "int"):
        """Initialize the encoder beforehand with a set of all possible values to encounter.

        Args:
            categories (list): List of possible values, matching the one-hot encoding.
            add_unknown (bool): Whether to add a unknown bit. Default is True.
        """
        assert isinstance(dtype, str)
        if dtype not in ["str", "int", "float"]:
            raise ValueError("Unsupported dtype for OneHotEncoder %s" % dtype)
        self.dtype_identifier = dtype
        self.dtype = self._dtype_translate[dtype]
        self.categories = [self.dtype(x) for x in categories]
        self.found_values = []
        self.add_unknown = add_unknown

    def __call__(self, value, **kwargs):
        r"""Encode a single feature or value, mapping it to a one-hot python list. E.g. `[0, 0, 1, 0]`

        Args:
            value: Any object that can be compared to items in ``self.one_hot_values``.
            **kwargs: Additional kwargs. Not used atm.

        Returns:
            list: Python list with 1 at value match. E.g. `[0, 0, 1, 0]`
        """
        encoded_list = [1 if x == self.dtype(value) else 0 for x in self.categories]
        if self.add_unknown:
            if value not in self.categories:
                encoded_list += [1]
            else:
                encoded_list += [0]
        if value not in self.found_values:
            self.found_values += [value]
        return encoded_list

    def get_config(self):
        config = {"categories": self.categories, "add_unknown": self.add_unknown, "dtype": self.dtype_identifier}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MolecularGraphRDKit(MolGraphInterface):
    """A graph object representing a strict molecular graph, e.g. only chemical bonds."""

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

    bond_fun_dict = {
        "BondType": lambda bond: bond.GetBondType(),
        "IsAromatic": lambda bond: bond.GetIsAromatic(),
        "IsConjugated": lambda bond: bond.GetIsConjugated(),
        "IsInRing": lambda bond: bond.IsInRing(),
        "Stereo": lambda bond: bond.GetStereo()
    }

    mol_fun_dict = {
        "ExactMolWt": rdkit.Chem.Descriptors.ExactMolWt,
        "NumAtoms": lambda mol_arg_lam: mol_arg_lam.GetNumAtoms()
    }

    def __init__(self, mol=None, add_hydrogen: bool = True, is_directed: bool = False,
                 make_conformers: bool = True):
        """Initialize MolecularGraphRDKit with mol object.

        Args:
            mol (rdkit.Chem.rdchem.Mol): Mol object from rdkit. Default is None.
            add_hydrogen (bool): Whether to add hydrogen. Default is True.
            is_directed (bool): Whether the edges are directed. Default is False.
            make_conformers (bool): Whether to make conformers. Default is True.
        """
        super().__init__(mol=mol, add_hydrogen=add_hydrogen)
        self.mol = mol
        self._add_hydrogen = add_hydrogen
        self.is_directed = is_directed
        self._make_conformers = make_conformers

    def make_conformers(self):
        if self.mol is None:
            return
        m = self.mol
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
                print("WARNING:kgcnn: Rdkit could not embed molecule with smile", m.GetProp("_Name"))
                return

    def from_smiles(self, smile, sanitize: bool = True):
        """Make molecule from smile.

        Args:
            smile (str): Smile string for the molecule.
            sanitize (bool): Whether to sanitize molecule.
        """
        # Make molecule from smile via rdkit
        m = rdkit.Chem.MolFromSmiles(smile)
        if m is None:
            print("ERROR:kgcnn: Rdkit can not convert smile %s" % smile)
            return self

        if sanitize:
            rdkit.Chem.SanitizeMol(m)
        if self._add_hydrogen:
            m = rdkit.Chem.AddHs(m)  # add H's to the molecule

        m.SetProp("_Name", smile)
        self.mol = m

        if self._make_conformers:
            self.make_conformers()

        return self

    @property
    def node_coordinates(self):
        m = self.mol
        if len(m.GetConformers()) > 0:
            return np.array(self.mol.GetConformers()[0].GetPositions())
        return []

    def to_smiles(self):
        if self.mol is None:
            return
        return rdkit.Chem.MolToSmiles(self.mol)

    def from_mol_block(self, mb, sanitize=False):
        if mb is None or len(mb) == 0:
            return self
        self.mol = rdkit.Chem.MolFromMolBlock(mb, removeHs=(not self._add_hydrogen), sanitize=sanitize)
        return self

    def to_mol_block(self):
        if self.mol is None:
            return None
        return rdkit.Chem.MolToMolBlock(self.mol)

    @property
    def node_symbol(self):
        return [x.GetSymbol() for x in self.mol.GetAtoms()]

    @property
    def node_number(self):
        return np.array([x.GetAtomicNum() for x in self.mol.GetAtoms()])

    @property
    def edge_number(self):
        return self.edge_attributes(["BondType"], encoder={"BondType": int})

    @property
    def edge_indices(self):
        m = self.mol
        bond_idx = []
        for i, x in enumerate(m.GetBonds()):
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
            if not self.is_directed:
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
        m = self.mol
        edges = self._check_properties_list(properties, sorted(self.bond_fun_dict.keys()), "Bond")
        encoder = self._check_encoder(encoder, sorted(self.bond_fun_dict.keys()))

        # Collect info about bonds
        bond_info = []
        bond_idx = []
        for i, x in enumerate(m.GetBonds()):
            attr = []
            for k in edges:
                temp = encoder[k](self.bond_fun_dict[k](x)) if k in encoder else [self.bond_fun_dict[k](x)]
                if isinstance(temp, (list, tuple)):
                    attr += list(temp)
                else:
                    attr.append(temp)
            bond_info.append(attr)
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])
            # Add a bond with opposite direction but same properties
            if not self.is_directed:
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
        m = self.mol
        nodes = self._check_properties_list(properties, sorted(self.atom_fun_dict.keys()), "Atom")
        encoder = self._check_encoder(encoder, sorted(self.atom_fun_dict.keys()))
        # Collect info about atoms
        atom_info = []
        for i, atm in enumerate(m.GetAtoms()):
            attr = []
            for k in nodes:
                temp = encoder[k](self.atom_fun_dict[k](atm)) if k in encoder else [self.atom_fun_dict[k](atm)]
                if isinstance(temp, (list, tuple)):
                    attr += list(temp)
                else:
                    attr.append(temp)
            atom_info.append(attr)
        return atom_info

    def graph_attributes(self, properties: list, encoder: dict):
        graph = self._check_properties_list(properties, sorted(self.mol_fun_dict.keys()), "Molecule")
        m = self.mol
        encoder = self._check_encoder(encoder, sorted(self.mol_fun_dict.keys()))
        # Mol info
        attr = []
        for k in graph:
            temp = encoder[k](self.mol_fun_dict[k](m)) if k in encoder else [self.mol_fun_dict[k](m)]
            if isinstance(temp, (list, tuple)):
                attr += list(temp)
            else:
                attr.append(temp)
        return attr
