import os
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from openbabel import openbabel
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# from openbabel import pybel


def single_smile_to_mol(smile: str,
                        sanitize: bool = True,
                        add_hydrogen: bool = True,
                        make_conformers: bool = True,
                        optimize_conformer: bool = True):
    """Workflow to make mol information, i.e. compute structure and conformation. With rdkit and openbabel only.

    Args:
        smile (str): List of smiles.

    Returns:
        list: list with mol strings.
    """
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
        return rdkit.Chem.MolToMolBlock(m)

    # Try openbabel next
    try:
        m = openbabel.OBMol()
        obconversion = openbabel.OBConversion()
        format_okay = obconversion.SetInAndOutFormats("smi", "mol")
        read_okay = obconversion.ReadString(m, smile)
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
        obconversion = None

    if m is not None:
        return obconversion.WriteString(m)

    print("WARNING: Openbabel and RDKit failed for %s." % smile)
    return None


def smile_to_mol_parallel(smile_list: list,
                          num_workers: int = None,
                          sanitize: bool = True,
                          add_hydrogen: bool = True,
                          make_conformers: bool = True,
                          optimize_conformer: bool = True):
    if num_workers is None:
        num_workers = os.cpu_count()

    # Default parallelize with rdkit and openbabel
    if num_workers == 1:
        mol_list = [single_smile_to_mol(x, sanitize=sanitize, add_hydrogen=add_hydrogen,
                                        make_conformers=make_conformers,
                                        optimize_conformer=optimize_conformer) for x in smile_list]
        return mol_list
    else:
        arg_list = [(x, sanitize, add_hydrogen, make_conformers, optimize_conformer) for x in smile_list]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            result = executor.map(single_smile_to_mol, *zip(*arg_list))
        mol_list = list(result)
        return mol_list


# # Former parallelization using multiprocessing
# if __name__ == '__main__':
#     # Input arguments from command line.
#     import multiprocessing as mp
#     import argparse
#     from distutils import util
#     from kgcnn.mol.io import read_smiles_file, write_mol_block_list_to_sdf
#     parser = argparse.ArgumentParser(description='Translate smiles.')
#     parser.add_argument("--file", required=True, help="Path to smile file")
#     parser.add_argument("--nprocs", required=False, help="nprocs", default=1, type=int)
#     parser.add_argument("--sanitize", required=False, help="sanitize", default=True, type=util.strtobool)
#     parser.add_argument("--add_hydrogen", required=False, help="add_hydrogen", default=True, type=util.strtobool)
#     parser.add_argument("--make_conformers", required=False, help="make_conformers", default=True, type=util.strtobool)
#     parser.add_argument("--optimize_conformer", required=False, help="optimize_conformer", default=True, type=util.strtobool)
#     args = vars(parser.parse_args())
#
#     arg_nprocs = args["nprocs"]
#     arg_file_path = args["file"]
#     arg_sanitize = args["sanitize"]
#     arg_add_hydrogen = args["add_hydrogen"]
#     arg_make_conformers = args["make_conformers"]
#     arg_optimize_conformer = args["optimize_conformer"]
#
#     smile_list = read_smiles_file(arg_file_path)
#     if arg_nprocs is None:
#         arg_nprocs = mp.cpu_count()
#
#     pool = mp.Pool(arg_nprocs)
#     results = pool.starmap(single_smile_to_mol,
#                            [(smile, arg_sanitize, arg_add_hydrogen, arg_make_conformers, arg_optimize_conformer)
#                             for smile in smile_list])
#     pool.close()
#
#     write_mol_block_list_to_sdf(results, os.path.splitext(arg_file_path)[0] + ".sdf")
