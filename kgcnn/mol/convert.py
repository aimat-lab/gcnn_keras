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
        _c_format_str = " {:.10f}"*len(c_iter) + "\n"
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


def parse_mol_str(mol_str: str, delimiter: str = None):
    """Parse a mol string into python nested list. Only supports V2000 format.

    Args:
        mol_str (str): String of mol block.
        delimiter (str): Delimiter for mol info. Default is None.

    Returns:
        list: list of bocks in mol-file. That is title, program, comment, counts, atoms, bonds, properties.
    """
    lines = mol_str.split("\n")
    lines = [x.strip().split(delimiter) for x in lines]
    lines = [[y for y in x if y != ''] for x in lines]
    lines[3] = [int(x) for x in lines[3][:-1]] + [lines[3][-1]]
    na = int(lines[3][0])
    nb = int(lines[3][1])
    out_list = []
    if lines[3][-1] == "V2000":
        # Separate into blocks:
        for x in lines[:4]:
            out_list.append(x)
        # atom block
        atoms = []
        for a in lines[4:(na + 4)]:
            # noinspection PyTypeChecker
            atoms.append([float(x) for x in a[:3]] + a[3:])
        out_list.append(atoms)
        # bond block
        bonds = []
        for b in lines[4 + na:4 + na + nb]:
            bonds.append([int(x) for x in b])
        out_list.append(bonds)
        # Properties block
        prop_block = []
        for p in lines[4 + na + nb:]:
            if len(p) >= 2:
                if p[0] == "M" and p[1] != "END":
                    prop_block.append(p)
        out_list.append(prop_block)
    else:
        raise NotImplementedError("ERROR:kgcnn: Can not parse mol V3000 or higher.")
    return out_list


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
    with open(filepath, "w+") as file:
        for i, mol_block in enumerate(mol_block_list):
            if mol_block is not None:
                file.write(mol_block)
                if i < len(mol_block_list)-1:
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
    with open(filepath, "r") as f:
        all_sting = f.read()
    return all_sting.split("$$$$\n")