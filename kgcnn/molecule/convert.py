import os
import logging
from typing import Callable
from kgcnn.molecule.io import read_mol_list_from_sdf_file, read_xyz_file, read_smiles_file, write_mol_block_list_to_sdf, \
    parse_list_to_xyz_str
from concurrent.futures import ThreadPoolExecutor  # , ProcessPoolExecutor
from kgcnn.molecule.external.ballloon import BalloonInterface
from typing import Union

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

# RDkit
try:
    import rdkit
    import rdkit.Chem
    import rdkit.Chem.AllChem
    import rdkit.Chem.rdDetermineBonds
    from rdkit import RDLogger

    def rdkit_smile_to_mol(smile: str, sanitize: bool = True, add_hydrogen: bool = True, make_conformers: bool = True,
                           optimize_conformer: bool = True, random_seed: int = 42, stop_logging: bool = False):
        # Order of parameters is important here.
        if stop_logging:
            RDLogger.DisableLog('rdApp.*')

        try:
            m = rdkit.Chem.MolFromSmiles(smile)
            if sanitize:
                rdkit.Chem.SanitizeMol(m)

            m = rdkit.Chem.AddHs(m)
            m.SetProp("_Name", smile.strip())

            if make_conformers:
                params = rdkit.Chem.AllChem.ETKDGv3()
                params.useSmallRingTorsions = True
                params.randomSeed = random_seed
                # params.useRandomCoords = True
                success = rdkit.Chem.AllChem.EmbedMolecule(m, params=params)
                if optimize_conformer:
                    rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)
                    rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
                    rdkit.Chem.AssignStereochemistryFrom3D(m)
            if not add_hydrogen:
                m = rdkit.Chem.RemoveHs(m)

            rdkit.Chem.AssignStereochemistry(m)

        except:
            m = None

        if stop_logging:
            RDLogger.EnableLog('rdApp.*')

        if m is not None:
            return rdkit.Chem.MolToMolBlock(m)

        return None

    def rdkit_xyz_to_mol(xyz_string: str, charge: Union[int, list, None] = None):
        """Convert xyz-string to mol-string.

        The order of atoms in the list should be the same as output.

        Args:
            xyz_string (str): Convert the xyz string to mol-string
            charge (int, list): Possible charges of the molecule.

        Returns:
            str: Mol-string. Generates bond information in addition to coordinates from xyz-string.
        """
        if charge is None:
            charge = [0, 1, -1, 2, -2]
        if isinstance(charge, int):
            charge = [charge]
        out_mol = None
        for c in charge:
            try:
                raw_mol = rdkit.Chem.MolFromXYZBlock(xyz_string)
                out_mol = rdkit.Chem.Mol(raw_mol)
                # rdkit.Chem.rdDetermineBonds.DetermineConnectivity(out_mol, charge=charge)
                rdkit.Chem.rdDetermineBonds.DetermineBonds(out_mol, charge=c)
                break
            except:
                out_mol = None
                continue
        if out_mol is not None:
            return rdkit.Chem.MolToMolBlock(out_mol)
        return None

except ImportError:
    module_logger.error("Can not import `RDKit` package for conversion.")
    rdkit_smile_to_mol = None
    rdkit_xyz_to_mol = None

try:
    # There problems with openbabel if system variable is not set.
    # Openbabel may not be fully threadsafe, but is improved in version 3.0.
    from openbabel import openbabel

    if "BABEL_DATADIR" not in os.environ:
        module_logger.warning(
            "In case openbabel fails, you can set `kgcnn.mol.convert.openbabel_smile_to_mol` to `None` for disable.")


    def openbabel_smile_to_mol(smile: str, sanitize: bool = True, add_hydrogen: bool = True,
                               make_conformers: bool = True, optimize_conformer: bool = True,
                               random_seed: int = 42,
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
                print(
                    "WARNING: Openbabel returned false flag %s" % [key for key, value in is_okay.items() if not value])
        except:
            m = None
            ob_conversion = None

        # Set back to default
        if stop_logging:
            openbabel.obErrorLog.StartLogging()

        if m is not None:
            return ob_conversion.WriteString(m)
        return None


    def openbabel_xyz_to_mol(xyz_string: str, charge: int = 0, stop_logging: bool = False):
        """Convert xyz-string to mol-string.

        The order of atoms in the list should be the same as output. Uses openbabel for conversion.

        Args:
            xyz_string (str): Convert the xyz string to mol-string
            stop_logging (bool): Whether to stop logging. Default is False.

        Returns:
            str: Mol-string. Generates bond information in addition to coordinates from xyz-string.
        """
        if stop_logging:
            openbabel.obErrorLog.StopLogging()

        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInAndOutFormats("xyz", "mol")
        # ob_conversion.SetInFormat("xyz")

        mol = openbabel.OBMol()
        ob_conversion.ReadString(mol, xyz_string)
        # print(xyz_str)

        out_mol = ob_conversion.WriteString(mol)

        # Set back to default
        if stop_logging:
            openbabel.obErrorLog.StartLogging()
        return out_mol

except ImportError:
    module_logger.error("Can not import `OpenBabel` package for conversion.")
    openbabel_smile_to_mol, openbabel_xyz_to_mol = None, None


class MolConverter:

    def __init__(self, base_path: str = None):
        """Initialize a converter to transform smile or coordinates into mol block information.

        Args:
            base_path (str): Base path for temporary files.
        """
        self.base_path = base_path

        if base_path is None:
            self.base_path = os.path.realpath(__file__)

    @staticmethod
    def _check_is_same_length(a, b):
        if len(a) != len(b):
            module_logger.error("Mismatch in number of converted. Found '%s' vs. '%s'." % (len(a), len(b)))
            raise ValueError("Conversion was not successful")

    @staticmethod
    def _convert_parallel(conversion_method: Callable, smile_list: list, num_workers: int, *args):
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

        module_logger.warning("Failed conversion for smile '%s'." % smile)
        return None

    def smile_to_mol(self, smiles_path: str, sdf_path: str, external_program: dict = None, num_workers: int = None,
                     sanitize: bool = True, add_hydrogen: bool = True, make_conformers: bool = True,
                     optimize_conformer: bool = True, logger=None, batch_size: int = 5000):
        """Convert a smiles file to SDF structure file.

        Args:
            smiles_path:
            sdf_path:
            external_program:
            num_workers:
            sanitize:
            add_hydrogen:
            make_conformers:
            optimize_conformer:
            logger:
            batch_size:

        Returns:
            list: List of mol-strings.
        """
        # Default via python packages RDkit and OpenBabel.
        if external_program is None:
            smiles_list = read_smiles_file(smiles_path)
            mol_list = []
            for i in range(0, len(smiles_list), batch_size):
                mg = self._convert_parallel(
                    self._single_smile_to_mol, smiles_list[i:i + batch_size], num_workers,
                    # All args for _single_smile_to_mol.
                    sanitize, add_hydrogen, make_conformers, optimize_conformer
                )
                mol_list = mol_list + mg
                if logger is not None:
                    logger.info(" ... converted molecules {0} from {1}".format(i + len(mg), len(smiles_list)))
            # Check success
            self._check_is_same_length(smiles_list, mol_list)
            if sdf_path is not None:
                write_mol_block_list_to_sdf(mol_list, sdf_path)
            return mol_list

        # External programs
        smiles_list = read_smiles_file(smiles_path)

        if external_program["class_name"] == "balloon":
            ext_program = BalloonInterface(**external_program["config"])
            ext_program.run(input_file=smiles_path, output_file=sdf_path, output_format="sdf")
        else:
            raise ValueError("Unknown program for conversion of smiles '%s'" % external_program)

        mol_list = read_mol_list_from_sdf_file(sdf_path)
        self._check_is_same_length(smiles_list, mol_list)
        return mol_list

    @staticmethod
    def _single_xyz_to_mol(xyz_string, charge=0):
        if rdkit_smile_to_mol is not None:
            mol = rdkit_xyz_to_mol(xyz_string, charge)
            if mol is not None:
                return mol

        if openbabel_smile_to_mol is not None:
            mol = openbabel_xyz_to_mol(xyz_string, charge)
            if mol is not None:
                return mol

        module_logger.warning("Failed conversion for xyz '%s'... ." % xyz_string[:20])
        return None

    def xyz_to_mol(self, xyz_path: str, sdf_path: str, charge: Union[list, int, None] = None):
        """Convert xyz info to structure file.

        Args:
            xyz_path:
            sdf_path:
            charge:

        Returns:
            list: List of mol blocks as string.
        """
        if openbabel_xyz_to_mol is None and rdkit_xyz_to_mol is None:
            raise ModuleNotFoundError("Can not convert XYZ to SDF format, missing package `OpenBabel` or `RDkit`.")

        xyz_list = read_xyz_file(xyz_path)
        mol_list = []
        for x in xyz_list:
            xyz_str = parse_list_to_xyz_str(x, number_coordinates=3)
            # No parallel conversion here, not necessary.
            mol_str = self._single_xyz_to_mol(xyz_str, charge=charge)
            mol_list.append(mol_str)
        self._check_is_same_length(xyz_list, mol_list)
        if sdf_path is not None:
            write_mol_block_list_to_sdf(mol_list, sdf_path)
        return mol_list
