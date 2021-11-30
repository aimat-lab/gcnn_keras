def convert_list_to_xyz_str(mol: list, comment: str = ""):
    """Convert list of atom and coordinates list into xyz-string.

    Args:
        mol (list): Tuple or list of `[['C', 'H', ...], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], ... ]]`.
        comment (str): Comment for comment line in xyz string. Default is "".

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
        _c_format_str = " {:.10f}" * len(c_iter) + "\n"
        xyz_str = xyz_str + _at_str + _c_format_str.format(*c_iter)
    return xyz_str


def write_list_to_xyz_file(filepath: str, mol_list: list):
    """Write a list of nested list of atom and coordinates into xyz-string. Uses :obj:`convert_list_to_xyz_str`.

    Args:
        filepath (str): Full path to file including name.
        mol_list (list): List of molecules, which is a list of pairs of atoms and coordinates of
            `[[['C', 'H', ... ], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], ... ]], ... ]`.
    """
    with open(filepath, "w+") as file:
        for x in mol_list:
            xyz_str = convert_list_to_xyz_str(x)
            file.write(xyz_str)


def parse_mol_str(mol_str: str):
    """Parse a MDL mol table string into nested list. Only supports V2000 format and CTab. Better rely on
    openbabel to do this. This function was a temporary solution.

    Args:
        mol_str (str): String of mol block.

    Returns:
        list: [title, program, comment, counts, atoms, bonds, properties]
    """
    empty_return = ["", "", "", [], [], [], []]
    if len(mol_str) == 0:
        print("ERROR: Received empty MLD mol-block string.")
        return empty_return
    lines = mol_str.split("\n")
    if len(lines) < 4:
        print("ERROR: Could not find counts line. Invalid format.")
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
            print("WARNING: No supporting atom lists (deprecated) or stext entries.")
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
            if "M" in p and p != "M  END":
                properties.append(p)
    else:
        raise NotImplementedError("ERROR: Can not parse mol V3000 or higher.")
    return [title, program, comment, counts, atoms, bonds, properties]


def read_xyz_file(file_path, delimiter: str = None):
    """Simple python script to read xyz-file and parse into a nested python list. Always returns a list with
    the geometries in xyz file.

    Args:
        file_path (str): Full path to xyz-file.
        delimiter (str): Delimiter for xyz separation. Default is ' '.

    Returns:
        list: Nested coordinates from xyz-file.
    """
    mol_list = []
    comment_list = []
    # open file
    infile = open(file_path, "r")
    lines = infile.readlines()
    # read separate entries
    file_pos = 0
    while file_pos < len(lines):
        line_list = lines[file_pos].strip().split(delimiter)
        line_list = [x.strip() for x in line_list if x != '']
        if len(line_list) == 1:
            num = int(line_list[0])
            atoms = []
            coordinates = []
            comment_list.append(lines[file_pos + 1].strip())
            for i in range(num):
                xyz_list = lines[file_pos + i + 2].strip().split(delimiter)
                xyz_list = [x.strip() for x in xyz_list if x != '']
                atoms.append(str(xyz_list[0]).lower().capitalize())
                coordinates.append([float(x) for x in xyz_list[1:]])
            mol_list.append([atoms, coordinates])
            file_pos += num + 2
        elif len(line_list) > 1:
            print("Mismatch in atoms and positions in xyz file %s" % file_path)
            file_pos += 1
        else:
            # Skip empty line is fine
            file_pos += 1
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


def dummy_load_sdf_file(filepath):
    """Simple loader to load a SDF file by only splitting.

    Args:
        filepath (str): File path for SDF file.

    Returns:
        list: List of mol blocks as string.
    """
    with open(filepath, "r") as f:
        all_sting = f.read()
    return all_sting.split("$$$$\n")
