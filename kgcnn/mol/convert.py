import os
import logging
import uuid
from openbabel import openbabel
from typing import Callable
from kgcnn.mol.io import read_mol_list_from_sdf_file, write_smiles_file
from concurrent.futures import ThreadPoolExecutor  # , ProcessPoolExecutor
from kgcnn.mol.external.ballloon import BalloonInterface

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


# RDkit
try:
    import rdkit
    import rdkit.Chem
    import rdkit.Chem.AllChem

    def rdkit_smile_to_mol(smile: str, sanitize: bool = True, add_hydrogen: bool = True, make_conformers: bool = True,
                           optimize_conformer: bool = True):
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

        return None

except ImportError:
    module_logger.error("Can not import `RDKit` package for conversion.")
    rdkit_smile_to_mol = None

try:
    # There problems with openbabel if system variable is not set.
    # Openbabel may not be fully threadsafe, but is improved in version 3.0.
    from openbabel import openbabel

    if "BABEL_DATADIR" not in os.environ:
        module_logger.warning(
            "In case openbabel fails, you can set `kgcnn.mol.convert.openbabel_smile_to_mol` to `None` for disable.")

    def convert_smile_to_mol_openbabel(smile: str, sanitize: bool = True, add_hydrogen: bool = True,
                                       make_conformers: bool = True, optimize_conformer: bool = True,
                                       stop_logging: bool = False):
        if stop_logging:
            openbabel.obErrorLog.StopLogging()

        try:
            m = openbabel.OBMol()
            ob_conversion = openbabel.OBConversion()
            format_okay = ob_conversion.SetInAndOutFormats("smi", "mol")
            read_okay = ob_conversion.ReadString(m, smile)
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
            ob_conversion = None

        # Set back to default
        if stop_logging:
            openbabel.obErrorLog.StartLogging()

        if m is not None:
            return ob_conversion.WriteString(m)
        return None

except ImportError:
    module_logger.error("Can not import `OpenBabel` package for conversion.")
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
        if rdkit_smile_to_mol is None and openbabel_smile_to_mol is None:
            raise ModuleNotFoundError("Can not convert smiles. Missing `RDkit` or `OpenBabel` packages.")

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
