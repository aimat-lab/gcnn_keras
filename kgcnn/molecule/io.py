import logging


def parse_list_to_xyz_str(mol: list, comment: str = "", number_coordinates: int = None):
    """Convert list of atom and coordinates list into xyz-string.

    Args:
        mol (list): Tuple or list of `[['C', 'H', ...], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], ... ]]`.
        comment (str): Comment for comment line in xyz string. Default is "".
        number_coordinates (int): Number of allowed coordinates.

    Returns:
        str: Information in xyz-string format.
    """
    atoms = mol[0]
    coordinates = mol[1]
    if len(atoms) != len(coordinates):
        raise ValueError("Number of atoms does not match number of coordinates for xyz string.")
    xyz_str = str(int(len(atoms))) + "\n"
    if "\n" in comment:
        raise ValueError("Line break must not be in the comment line for xyz string.")
    xyz_str = xyz_str + comment + "\n"
    for a_iter, c_iter in zip(atoms, coordinates):
        _at_str = str(a_iter)
        if number_coordinates is not None:
            c_iter = c_iter[:number_coordinates]
        _c_format_str = " {:.10f}" * len(c_iter) + "\n"
        xyz_str = xyz_str + _at_str + _c_format_str.format(*c_iter)
    return xyz_str


def write_list_to_xyz_file(filepath: str, mol_list: list):
    """Write a list of nested list of atom and coordinates into xyz-string. Uses :obj:`parse_list_to_xyz_str`.

    Args:
        filepath (str): Full path to file including name.
        mol_list (list): List of molecules, which is a list of pairs of atoms and coordinates of
            `[[['C', 'H', ... ], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], ... ]], ... ]`.
    """
    with open(filepath, "w+") as file:
        for x in mol_list:
            xyz_str = parse_list_to_xyz_str(x)
            file.write(xyz_str)


def parse_mol_str(mol_str: str):
    """Parse MDL mol table string into nested list. Only supports V2000 format and CTab. Better rely on
    OpenBabel to do this. This function was a temporary solution.

    Args:
        mol_str (str): String of mol block.

    Returns:
        list: [title, program, comment, counts, atoms, bonds, properties]
    """
    empty_return = ["", "", "", [], [], [], []]
    if len(mol_str) == 0:
        logging.error("Received empty MLD mol-block string. Nothing to parse. Return empty list.")
        return empty_return
    lines = mol_str.split("\n")
    if len(lines) < 4:
        logging.error("Could not find counts line. Invalid format. Can not parse string. Return empty list.")
        return empty_return

    title = lines[0]
    program = lines[1]  # IIPPPPPPPPMMDDYYHHmmddSSssssssssssEEEEEEEEEEEERRRRRR
    comment = lines[2]
    version = lines[3][-6:].strip()
    if version == "V2000":
        # counts has aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
        # or shorter but should have version of len=5 at the end
        counts = [lines[3][i:i + 3].strip() for i in range(0, len(lines[3][:-6]), 3)] + [version]
        na = int(counts[0])
        nb = int(counts[1])
        nl = int(counts[2])
        ns = int(counts[5])
        if ns != 0 or nl != 0:
            logging.warning("Not supporting atom lists (deprecated) or stext entries for this function.")
        atoms = []
        for a in lines[4:(na + 4)]:
            # xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaaddcccssshhhbbbvvvHHHrrriiimmmnnneee
            # noinspection PyTypeChecker
            atoms.append([a[0:10].strip(), a[10:20].strip(), a[20:30].strip(), a[30:34].strip(), a[34:36].strip(),
                          a[36:39].strip(), a[39:42].strip(), a[42:45].strip(), a[45:48].strip(), a[48:51].strip(),
                          a[51:54].strip(), a[54:57].strip(), a[57:60].strip(), a[60:63].strip(), a[63:66].strip(),
                          a[66:69].strip()])
        # bond block
        bonds = []
        for b in lines[4 + na:4 + na + nb]:
            # 111222tttsssxxxrrrccc
            # noinspection PyTypeChecker
            bonds.append([b[i:i+3].strip() for i in range(0, len(b), 3)])
        # Properties block
        properties = []
        for p in lines[4 + na + nb:]:
            if p == "M  END":
                break
            if "M" in p:
                properties.append(p)
    else:
        raise NotImplementedError("Can not parse mol V3000 or higher.")
    return [title, program, comment, counts, atoms, bonds, properties]


def read_xyz_file(file_path, delimiter: str = None, line_by_line=False):
    """Simple python script to read xyz-file and parse into a nested python list. Always returns a list with
    the geometries in xyz file.

    Args:
        file_path (str): Full path to xyz-file.
        delimiter (str): Delimiter for xyz separation. Default is ' '.
        line_by_line (bool): Whether to read XYZ file line by line.

    Returns:
        list: Nested coordinates from xyz-file.
    """
    mol_list = []
    comment_list = []
    # open file
    infile = open(file_path, "r")
    if line_by_line:
        lines = infile  # File object
    else:
        lines = infile.readlines()  # list of lines

    num = 0
    comment = 0
    atoms = []
    coordinates = []
    for line in lines:
        line_list = line.strip().split(delimiter)
        line_list = [x.strip() for x in line_list if x != ""]  # Remove multiple delimiter
        if len(line_list) == 1 and num == 0 and comment == 0:
            # Start new conformer and set line counts to read.
            num = int(line_list[0])
            comment = 1
        elif comment > 0:
            # Comment comes before atom block and must always be read.
            comment_list.append(str(line))
            comment = 0
        elif num > 0:
            if len(line_list) <= 1:
                logging.error("Expected to read atom-coordinate block but got comment or line count instead.")
            atoms.append(str(line_list[0]).lower().capitalize())
            coordinates.append([float(x) for x in line_list[1:]])
            if num == 1:
                # This was last line for this conformer. Append result and reset current list.
                mol_list.append([atoms, coordinates])
                num = 0
                atoms = []
                coordinates = []
            else:
                # Finished reading an atom line.
                num = num - 1
        else:
            logging.warning("Empty line in xyz file for mismatch in atom count found.")
    # close file
    infile.close()
    return mol_list


def write_mol_block_list_to_sdf(mol_block_list, filepath):
    """Write a list of mol blocks as string into a SDF file.

    Args:
        mol_block_list (list): List of mol blocks as string.
        filepath (str): File path for SDF file.

    Returns:
        None.
    """
    with open(filepath, "w+") as file:
        for i, mol_block in enumerate(mol_block_list):
            if mol_block is not None:
                file.write(mol_block)
                if i < len(mol_block_list) - 1:
                    file.write("$$$$\n")
            else:
                file.write("".join(["\n",
                                    "     FAIL\n",
                                    "\n",
                                    "  0  0  0  0  0  0  0  0  0  0 V2000\n",
                                    "M  END\n"]))
                if i < len(mol_block_list) - 1:
                    file.write("$$$$\n")


def read_mol_list_from_sdf_file(filepath, line_by_line=False):
    """Simple loader to load an SDF file by only splitting.

    Args:
        filepath (str): File path for SDF file.
        line_by_line (bool): Whether to read SDF file line by line.

    Returns:
        list: List of mol blocks as string.
    """
    mol_list = []
    with open(filepath, "r") as f:
        if not line_by_line:
            all_sting = f.read()
            mol_list = all_sting.split("$$$$\n")
        else:
            iter_mol = ""
            for line in f:
                if line == "$$$$\n":
                    mol_list.append(iter_mol)
                    iter_mol = ""
                else:
                    iter_mol = iter_mol + line
            if iter_mol != "":
                mol_list.append(iter_mol)
    # Check if there was tailing $$$$ with nothing to follow.
    # Split will make empty string at the end, which does not match actual number of mol blocks.
    if len(mol_list[-1]) == 0:
        mol_list = mol_list[:-1]
    return mol_list


def read_smiles_file(file_path):
    """Simply python function to read smiles from file.

    Args:
        file_path (str): File path for smiles file.

    Returns:
        list: List of smiles.
    """
    with open(file_path, "r") as f:
        smile_list = [line.rstrip() for line in f]
    return smile_list


def write_smiles_file(file_path, smile_list):
    """Simply python function to write smiles to file.

    Args:
        file_path (str): File path for smiles file.
        smile_list (list): List of smiles to write to file.

    Returns:
        None
    """
    with open(file_path, "w+") as f:
        for i, x in enumerate(smile_list):
            if i == len(smile_list)-1:
                f.write(x)
            else:
                f.write(x + "\n")
