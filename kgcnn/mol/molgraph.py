import numpy as np

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.Descriptors

# For this module please install rdkit. See: https://www.rdkit.org/docs/Install.html
# or check https://pypi.org/project/rdkit-pypi/

class OneHotEncoder:

    def __init__(self, onehot_values, add_others=True):
        self.onehot_values = onehot_values
        self.found_values = []
        self.add_others = add_others

    def __call__(self, value, **kwargs):
        encoded_list = [1 if x == value else 0 for x in self.onehot_values]
        if self.add_others:
            if value not in self.onehot_values:
                encoded_list += [1]
            else:
                encoded_list += [0]
        if value not in self.found_values:
            self.found_values += [value]
        return encoded_list


class MolecularGraph:

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

    def __init__(self, smile, add_Hs=True, sanitize=True):
        """Make molecule from smile.

        Args:
            smile (str): Smile string for the molecule.
            add_Hs (bool): Whether to add hydrogen after creating the smile mol object. Default is True.
        """

        # Make molecule from smile via rdkit
        m = rdkit.Chem.MolFromSmiles(smile)
        if sanitize:
            rdkit.Chem.SanitizeMol(m)
        if add_Hs:
            m = rdkit.Chem.AddHs(m)  # add H's to the molecule

        try:
            # rdkit.Chem.FindPotentialStereo(m)  # Assign Stereochemistry new method
            rdkit.Chem.AssignStereochemistry(m)
            rdkit.Chem.AllChem.EmbedMolecule(m, useRandomCoords=True)
            rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)
            # rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
            rdkit.Chem.AssignStereochemistryFrom3D(m)
            rdkit.Chem.AssignStereochemistry(m)
        except ValueError:
            try:
                rdkit.Chem.RemoveStereochemistry(m)
                rdkit.Chem.AssignStereochemistry(m)
                rdkit.Chem.AllChem.EmbedMolecule(m, useRandomCoords=True)
                rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)
                # rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
                rdkit.Chem.AssignStereochemistryFrom3D(m)
                rdkit.Chem.AssignStereochemistry(m)
            except ValueError:
                print("WARNING: Rdkit could not embed molecule with smile", smile)



        self.mol = m
        self.atom_labels = None
        self.atom_features = None
        self.bond_indices = None
        self.bond_features = None
        self.coordinates = None
        self.molecule_features = None

    def make_features(self, nodes=None, edges=None, state=None, encoder=None, is_directed=True):
        """This is a rudiment function for converting a smile string to a graph-like object.

        Args:
            nodes (list): Optional list of node properties to extract.
            edges (list): Optional list of edge properties to extract.
            state (list): Optional list of molecule properties to extract.
            is_directed (bool): Whether to add a bond for a directed graph. Default is True.
            encoder (dict): Callable object for property to return list. Example: OneHotEncoder. Default is None.

        Returns:
            self: MolecularGraph object.
        """

        m = self.mol

        # Define the properties to extract
        if nodes is None:
            nodes = sorted(self.atom_fun_dict.keys())
        else:
            nodes_unknown = [x for x in nodes if x not in sorted(self.atom_fun_dict.keys())]
            if len(nodes_unknown) > 0:
                print("WARNING: Atom property is not defined, ignore following keys:", nodes_unknown)
            nodes = [x for x in nodes if x in sorted(self.atom_fun_dict.keys())]
        if edges is None:
            edges = sorted(self.bond_fun_dict.keys())
        else:
            edges_unknown = [x for x in edges if x not in sorted(self.bond_fun_dict.keys())]
            if len(edges_unknown) > 0:
                print("WARNING: Bond property is not defined, ignore following keys:", edges_unknown)
            edges = [x for x in edges if x in sorted(self.bond_fun_dict.keys())]
        if state is None:
            state = sorted(self.mol_fun_dict.keys())
        else:
            state_unknown = [x for x in state if x not in sorted(self.mol_fun_dict.keys())]
            if len(state_unknown) > 0:
                print("WARNING: Molecule property is not defined, ignore following keys:", state_unknown)
            state = [x for x in state if x in sorted(self.mol_fun_dict.keys())]

        # Encoder
        if encoder is None:
            encoder = {}
        else:
            encoder_unknown = [x for x in encoder if x not in sorted(self.atom_fun_dict.keys()) and x not in sorted(
                self.bond_fun_dict.keys()) and x not in sorted(self.mol_fun_dict.keys())]
            if len(encoder_unknown) > 0:
                print("WARNING: Encoder property not known", encoder_unknown)
            encoder = {key: value for key, value in encoder.items() if key not in encoder_unknown}

        # List of information and properties
        self.atom_labels = [rdkit.Chem.rdchem.Atom.GetSymbol(x) for x in m.GetAtoms()]
        if len(m.GetConformers()) > 0:
            self.coordinates = m.GetConformers()[0].GetPositions()

        # Features to fill
        atom_info = []
        bond_idx = []
        bond_info = []
        mol_info = []

        # Mol info
        for k in state:
            mol_info += encoder[k](self.mol_fun_dict[k](m)) if k in encoder else [self.mol_fun_dict[k](m)]

        # Collect info about atoms
        for i, atm in enumerate(m.GetAtoms()):
            attr = []
            for k in nodes:
                attr += encoder[k](self.atom_fun_dict[k](atm)) if k in encoder else [self.atom_fun_dict[k](atm)]
            atom_info.append(attr)

        # Collect info about bonds
        for i, x in enumerate(m.GetBonds()):
            attr = []
            for k in edges:
                attr += encoder[k](self.bond_fun_dict[k](x)) if k in encoder else [self.bond_fun_dict[k](x)]
            bond_info.append(attr)
            bond_idx.append([x.GetBeginAtomIdx(), x.GetEndAtomIdx()])

            # Add a bond with opposite direction but same properties
            if is_directed:
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

        self.atom_features = atom_info
        self.molecule_features = mol_info
        self.bond_indices = bond_idx
        self.bond_features = bond_info

        return self
