import pymatgen
import pymatgen.io.cif


def parse_cif_file_to_structures(cif_file: str):
    # structure = pymatgen.io.cif.CifParser.from_string(cif_string).get_structures()[0]
    structures = pymatgen.io.cif.CifParser(cif_file).get_structures()
    return structures


def convert_structures_as_dict(structures: list):
    dicts = [s.as_dict() for s in structures]
    return dicts

