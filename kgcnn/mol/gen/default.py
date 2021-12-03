import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from openbabel import openbabel
# from openbabel import pybel

# Molecule and Conformer generation with python interface tools RdKit, OpenBabel


def smile_to_mol(smile_list: list,
                 sanitize: bool = True,
                 add_hydrogen: bool = True,
                 make_conformers: bool = True,
                 optimize_conformer: bool = True):
    """Workflow to make mol information, i.e. compute structure and conformation. With rdkit and openbabel only.

    Args:
        smile_list (list): List of smiles.

    Returns:
        list: list with mol strings.
    """
    mol_list = []
    for smile in smile_list:
        # Try rdkit first
        try:
            m = rdkit.Chem.MolFromSmiles(smile)
            if sanitize:
                rdkit.Chem.SanitizeMol(m)
            if add_hydrogen:
                m = rdkit.Chem.AddHs(m)  # add H's to the molecule
            m.SetProp("_Name", smile)
            if make_conformers:
                rdkit.Chem.RemoveStereochemistry(m)
                rdkit.Chem.AssignStereochemistry(m)
                rdkit.Chem.AllChem.EmbedMolecule(m, useRandomCoords=True)
            if optimize_conformer and make_conformers:
                rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)
                rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
                rdkit.Chem.AssignStereochemistryFrom3D(m)
                rdkit.Chem.AssignStereochemistry(m)
        except:
            m = None

        if m is not None:
            mol_list.append(rdkit.Chem.MolToMolBlock(m))
            continue

        # Try openbabel next
        try:
            m = openbabel.OBMol()
            obconversion = openbabel.OBConversion()
            format_okay = obconversion.SetInAndOutFormats("smi", "mol")
            read_okay = obconversion.ReadString(m, smile)
            is_okay = [format_okay, read_okay]
            if make_conformers:
                # We need to make conformer with builder
                builder = openbabel.OBBuilder()
                build_okay = builder.Build(m)
                is_okay.append(build_okay)
            if add_hydrogen:
                # it seems h's are made after build, an get embedded too
                m.AddHydrogens()
            if optimize_conformer and make_conformers:
                ff = openbabel.OBForceField.FindType("mmff94")
                ffsetup_okay = ff.Setup(m)
                ff.SteepestDescent(50)  # 50 steps here
                ff.GetCoordinates(m)
                is_okay.append(ffsetup_okay)
            all_okay = all(is_okay)
            if not all_okay:
                print("WARNING: Openbabel returned false flag at %s" % is_okay)
        except:
            m = None
            obconversion = None

        if m is not None:
            mol_list.append(obconversion.WriteString(m))
            continue

        print("WARNING: Openbabel and RDKit failed for %s." % smile)

        # Nothing worked
        mol_list.append(None)
    return mol_list
