import os
import sys
import subprocess
import uuid

from kgcnn.mol.io import dummy_load_sdf_file, write_smiles_file


def run_default(file_path,
                nprocs: int = None,
                sanitize: bool = True,
                add_hydrogen: bool = True,
                make_conformers: bool = True,
                optimize_conformer: bool = True):
    if sys.platform[0:3] == 'win':
        python_command = 'python'  # or 'python.exe'
    else:
        python_command = 'python3'
    script_command = os.path.join(os.path.dirname(os.path.realpath(__file__)), "default.py")
    command_list = [python_command, script_command,
                    "--file", file_path,
                    "--nprocs", str(nprocs),
                    "--sanitize", str(sanitize),
                    "--add_hydrogen", str(add_hydrogen),
                    "--make_conformers", str(make_conformers),
                    "--optimize_conformer", str(optimize_conformer)
                    ]
    return command_list


def smile_to_mol(smile_list: list,
                 base_path: str = None,
                 conv_program: str = "default",
                 nprocs: int = None,
                 sanitize: bool = True,
                 add_hydrogen: bool = True,
                 make_conformers: bool = True,
                 optimize_conformer: bool = True):
    """Workflow to make mol information, i.e. compute structure and conformation. With rdkit and openbabel only.

    Args:
        smile_list (list): List of smiles.
        base_path: None
        conv_program: "default"
        nprocs: None
        sanitize: True
        add_hydrogen: True
        make_conformers: True
        optimize_conformer: True

    Returns:
        list: list with mol strings.
    """
    if base_path is None:
        base_path = os.path.realpath(__file__)
    if nprocs is None:
        nprocs = os.cpu_count()

    smile_file = os.path.join(base_path, str(uuid.uuid4()) + ".smile")
    write_smiles_file(smile_file, smile_list)

    if conv_program == "default":
        return_code = subprocess.run(run_default(file_path=smile_file,
                                                 nprocs=nprocs,
                                                 sanitize=sanitize,
                                                 add_hydrogen=add_hydrogen,
                                                 make_conformers=make_conformers,
                                                 optimize_conformer=optimize_conformer))
    else:
        raise ValueError("Unknown program for conversion of smiles %s" % conv_program)
    # Check return code
    if int(return_code.returncode) != 0:
        raise ValueError("Batch process returned with error:", return_code)
    else:
        # print("Batch process returned:", return_code)
        pass

    mol_file = os.path.splitext(smile_file)[0] + ".sdf"
    mol_list = dummy_load_sdf_file(mol_file)

    if len(smile_list) != len(mol_list):
        print("Mismatch in number of converted smiles. That is %s vs. %s" % (len(smile_list), len(mol_list)))
        raise ValueError("Smile conversion was not successful")

    # Clean up
    os.remove(mol_file)
    os.remove(smile_file)

    return mol_list
