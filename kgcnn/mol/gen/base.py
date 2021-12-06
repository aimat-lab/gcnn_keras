import os
import sys
import uuid

from kgcnn.mol.io import dummy_load_sdf_file, write_smiles_file
from kgcnn.mol.gen.default import smile_to_mol_parallel


def smile_to_mol(smile_list: list,
                 base_path: str = None,
                 conv_program: dict = None,
                 num_workers: int = None,
                 sanitize: bool = True,
                 add_hydrogen: bool = True,
                 make_conformers: bool = True,
                 optimize_conformer: bool = True):
    """Workflow to make mol information, i.e. compute structure and conformation.

    Args:
        smile_list (list): List of smiles.
        base_path: None
        conv_program: "default"
        num_workers: None
        sanitize: True
        add_hydrogen: True
        make_conformers: True
        optimize_conformer: True

    Returns:
        list: list with mol strings.
    """
    if base_path is None:
        base_path = os.path.realpath(__file__)
    if num_workers is None:
        num_workers = os.cpu_count()

    def check_is_correct_length(a, b):
        if len(a) != len(b):
            print("Mismatch in number of converted. That is %s vs. %s" % (len(a), len(b)))
            raise ValueError("Conversion was not successful")

    if conv_program is None:
        # Default via rdkit and openbabel
        mol_list = smile_to_mol_parallel(smile_list=smile_list,
                                         num_workers=num_workers,
                                         sanitize=sanitize,
                                         add_hydrogen=add_hydrogen,
                                         make_conformers=make_conformers,
                                         optimize_conformer=optimize_conformer)
        # Check success
        check_is_correct_length(smile_list, mol_list)
        return mol_list

    # External programs
    # Write out temporary smiles file.
    smile_file = os.path.join(base_path, str(uuid.uuid4()) + ".smile")
    write_smiles_file(smile_file, smile_list)

    if conv_program == "balloon":
        pass
    else:
        raise ValueError("Unknown program for conversion of smiles %s" % conv_program)

    mol_file = os.path.splitext(smile_file)[0] + ".sdf"
    mol_list = dummy_load_sdf_file(mol_file)
    # Clean up
    os.remove(mol_file)
    os.remove(smile_file)

    # Check success
    check_is_correct_length(smile_list, mol_list)
    return mol_list
