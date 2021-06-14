import numpy as np

try:
    import rdkit
    import rdkit.Chem
    import rdkit.Chem.AllChem
except ModuleNotFoundError:
    print("Error: For this module please install rdkit. See: https://www.rdkit.org/docs/Install.html")

import rdkit
import rdkit.Chem
import rdkit.Chem.Descriptors
import rdkit.Chem.AllChem


# Will be replaced by a more general method in the future.
def smile_to_graph(smile, nodes=None, edges=None, state=None, is_directed=True):
    """This is a rudiment function for converting a smile string to a graph-like object."""

    atom_fun_dict = {
        "AtomicNum": rdkit.Chem.rdchem.Atom.GetAtomicNum,
        # "Symbol": rdkit.Chem.rdchem.Atom.GetSymbol,
        "NumExplicitHs": rdkit.Chem.rdchem.Atom.GetNumExplicitHs,
        "NumImplicitHs": rdkit.Chem.rdchem.Atom.GetNumImplicitHs,
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
        "Idx": rdkit.Chem.rdchem.Atom.GetIdx
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

    # Define the properties to extract
    if nodes is None:
        nodes = sorted(atom_fun_dict.keys())
    else:
        nodes_unknown = [x for x in nodes if x not in sorted(atom_fun_dict.keys())]
        if len(nodes_unknown)>0:
            print("Warning: Atom property is not defined, ignore following keys:", nodes_unknown)
        nodes = [x for x in nodes if x in sorted(atom_fun_dict.keys())]
    if edges is None:
        edges = sorted(bond_fun_dict.keys())
    else:
        edges_unknown = [x for x in edges if x not in sorted(bond_fun_dict.keys())]
        if len(edges_unknown) > 0:
            print("Warning: Bond property is not defined, ignore following keys:", edges_unknown)
        edges = [x for x in edges if x in sorted(bond_fun_dict.keys())]
    if state is None:
        state = sorted(mol_fun_dict.keys())
    else:
        state_unknown = [x for x in state if x not in sorted(mol_fun_dict.keys())]
        if len(state_unknown) > 0:
            print("Warning: Molecule property is not defined, ignore following keys:", state_unknown)
        state = [x for x in state if x in sorted(mol_fun_dict.keys())]


    # Make molecule from smile via rdkit
    m = rdkit.Chem.MolFromSmiles(smile)
    m = rdkit.Chem.AddHs(m)  # add H's to the molecule
    rdkit.Chem.AssignStereochemistry(m)
    # rdkit.Chem.FindPotentialStereo(m)  # Assign Stereochemistry new method
    rdkit.Chem.AllChem.EmbedMolecule(m)
    rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)

    # List of information and properties
    atom_sym = [rdkit.Chem.rdchem.Atom.GetSymbol(x) for x in m.GetAtoms()]
    atom_pos = m.GetConformers()[0].GetPositions()
    atom_info = []
    bond_idx = []
    bond_info = []
    mol_info = [mol_fun_dict[k](m) for k in state]

    # Collect info about atoms
    for i, atm in enumerate(m.GetAtoms()):
        attr = [atom_fun_dict[k](atm) for k in nodes]
        atom_info.append(attr)

    # Collect info about bonds
    for i, x in enumerate(m.GetBonds()):
        attr = [bond_fun_dict[k](x) for k in edges]
        bond_info.append(attr)
        bond_idx.append([x.GetBeginAtomIdx(), x.GetEndAtomIdx()])

        # Add a bond with opposite direction but same properties
        if is_directed:
            bond_info.append(attr)
            bond_idx.append([x.GetEndAtomIdx(), x.GetBeginAtomIdx()])

    # Sort directed bonds
    bond_idx = np.array(bond_idx, dtype="int64")
    order1 = np.argsort(bond_idx[:, 1], axis=0, kind='mergesort')  # stable!
    ind1 = bond_idx[order1]
    val1 = [bond_info[i] for i in order1]
    order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')  # stable!
    ind2 = ind1[order2]
    val2 = [val1[i] for i in order2]

    # Take the sorted bonds
    bond_idx = ind2
    bond_info = val2

    return atom_sym, atom_pos, atom_info, bond_idx, bond_info, mol_info
