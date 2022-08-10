import os
import logging
import uuid
from typing import Callable
from concurrent.futures import ThreadPoolExecutor  # ,ProcessPoolExecutor
from kgcnn.mol.external.ballloon import BalloonInterface
from kgcnn.mol.io import read_mol_list_from_sdf_file, write_smiles_file

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

try:
    from kgcnn.mol.module_rdkit import convert_smile_to_mol_rdkit as rdkit_smile_to_mol
except ImportError:
    module_logger.error("Can not import RDKit module for conversion.")
    rdkit_smile_to_mol = None

try:
    from kgcnn.mol.module_babel import convert_smile_to_mol_openbabel as openbabel_smile_to_mol
    # There are problems with openbabel if system variable is not set.
    # Openbabel may not be fully threadsafe, but is improved in version 3.0.
    if "BABEL_DATADIR" not in os.environ:
        module_logger.warning(
            "In case openbabel fails, you can set `kgcnn.mol.convert.openbabel_smile_to_mol` to `None` for disable.")
except ImportError:
    module_logger.error("Can not import OpenBabel module for conversion.")
    openbabel_smile_to_mol = None


class MolConverter:

    def __init__(self,
                 base_path: str = None,
                 external_program: dict = None,
                 num_workers: int = None,
                 sanitize: bool = True,
                 add_hydrogen: bool = True,
                 make_conformers: bool = True,
                 optimize_conformer: bool = True):
        """Initialize a converter to transform smile or coordinates into mol block information.

        Args:
            base_path (str):
            external_program (dict):
            num_workers (int):
            sanitize (bool):
            add_hydrogen (bool):
            make_conformers (bool):
            optimize_conformer (bool):
        """
        self.base_path = base_path
        self.external_program = external_program
        self.num_workers = num_workers
        self.sanitize = sanitize
        self.add_hydrogen = add_hydrogen
        self.make_conformers = make_conformers
        self.optimize_conformer = optimize_conformer

        if base_path is None:
            self.base_path = os.path.realpath(__file__)
        if num_workers is None:
            self.num_workers = os.cpu_count()

    @staticmethod
    def _check_is_correct_length(a, b):
        if len(a) != len(b):
            module_logger.error("Mismatch in number of converted. Found %s vs. %s" % (len(a), len(b)))
            raise ValueError("Conversion was not successful")

    @staticmethod
    def _convert_parallel(conversion_method: Callable,
                          smile_list: list,
                          num_workers: int,
                          *args
                          ):
        if num_workers is None:
            num_workers = os.cpu_count()

        if num_workers == 1:
            mol_list = [conversion_method(x, *args) for x in smile_list]
            return mol_list
        else:
            arg_list = [(x,) + args for x in smile_list]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                result = executor.map(conversion_method, *zip(*arg_list))
            mol_list = list(result)
            return mol_list

    @staticmethod
    def _single_smile_to_mol(smile: str,
                             sanitize: bool = True,
                             add_hydrogen: bool = True,
                             make_conformers: bool = True,
                             optimize_conformer: bool = True):
        if rdkit_smile_to_mol is not None:
            mol = rdkit_smile_to_mol(smile=smile, sanitize=sanitize, add_hydrogen=add_hydrogen,
                                     make_conformers=make_conformers, optimize_conformer=optimize_conformer)
            if mol is not None:
                return mol

        if openbabel_smile_to_mol is not None:
            mol = openbabel_smile_to_mol(smile=smile, sanitize=sanitize, add_hydrogen=add_hydrogen,
                                         make_conformers=make_conformers, optimize_conformer=optimize_conformer)
            if mol is not None:
                return mol

        module_logger.warning("Failed conversion for smile %s" % smile)
        return None

    def smile_to_mol(self, smile_list: list):

        if self.external_program is None:
            # Default via rdkit and openbabel
            mol_list = self._convert_parallel(self._single_smile_to_mol,
                                              smile_list,
                                              self.num_workers,
                                              self.sanitize,
                                              self.add_hydrogen,
                                              self.make_conformers,
                                              self.optimize_conformer)
            # Check success
            self._check_is_correct_length(smile_list, mol_list)
            return mol_list

        # External programs

        # Write out temporary smiles file.
        smile_file = os.path.join(self.base_path, str(uuid.uuid4()) + ".smi")
        mol_file = os.path.splitext(smile_file)[0] + ".sdf"

        write_smiles_file(smile_file, smile_list)

        if self.external_program["class_name"] == "balloon":
            ext_program = BalloonInterface(**self.external_program["config"])
            ext_program.run(input_file=smile_file, output_file=mol_file, output_format="sdf")
        else:
            raise ValueError("Unknown program for conversion of smiles %s" % self.external_program)

        mol_list = read_mol_list_from_sdf_file(mol_file)
        # Clean up
        os.remove(mol_file)
        os.remove(smile_file)

        # Check success
        self._check_is_correct_length(smile_list, mol_list)
        return mol_list
