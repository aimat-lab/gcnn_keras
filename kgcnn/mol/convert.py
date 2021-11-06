def convert_list_to_xyz_str(atoms: list):
    """Convert nested list of atom and coordinates list into xyz-string.

    Args:
        atoms (list): Atom list of type `[['H', 0.0, 0.0, 0.0], ['C', 1.0, 1.0, 1.0], ...]`.

    Returns:
        str: Information in xyz-string format.
    """
    xyz_str = str(int(len(atoms))) + "\n"
    for a_iter in atoms:
        xyz_str = xyz_str + "\n"
        _line_str = "{:} {:.10f} {:.10f} {:.10f}".format(*a_iter)
        xyz_str = xyz_str + _line_str
    return xyz_str


def convert_xyz_to_mol_ob(xyz_str: str, stop_logging: bool = True):
    """Conversion of xyz-string to mol-string.

    The order of atoms in the list should be the same as output. Uses openbabel for conversion.

    Args:
        xyz_str (str): Convert the xyz string to mol-string
        stop_logging (bool): Whether to stop logging. Default is True.
    Returns:
        str: Mol-string from xyz-information. Generates structure or bond information.
    """
    try:
        from openbabel import openbabel
    except ImportError:
        raise ImportError("ERROR:kgcnn: Conversion from xyz to mol requires openbabel. Please install openbabel")

    if stop_logging:
        openbabel.obErrorLog.StopLogging()

    ob_conversion = openbabel.OBConversion()
    ob_conversion.SetInAndOutFormats("xyz", "mol")
    # ob_conversion.SetInFormat("xyz")

    mol = openbabel.OBMol()
    ob_conversion.ReadString(mol, xyz_str)
    # print(xyz_str)

    out_mol = ob_conversion.WriteString(mol)

    # Set back to default
    if stop_logging:
        openbabel.obErrorLog.StartLogging()
    return out_mol


def parse_mol_str(mol_str: str, delimiter: str = " "):
    """Parse a mol string into python nested list.

    Only supports V2000 format.

    Args:
        mol_str (str):
        delimiter (str):

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


def read_xyz_file(filepath, delimiter: str = " "):
    """Simple python script to read xyz-file and parse into a nested python list.

    Args:
        filepath (str): Full path to xyz-file.
        delimiter (str): Delimiter for xyz separation. Default is ' '.

    Returns:
        list: Nested coordinates from xyz-file.
    """
    mol_list = []
    comment_list = []
    # open file
    infile = open(filepath, "r")
    lines = infile.readlines()
    # read separate entries
    file_pos = 0
    while file_pos < len(lines):
        line_list = lines[file_pos].strip().split(delimiter)
        line_list = [x.strip() for x in line_list if x != '']
        if len(line_list) == 1:
            num = int(line_list[0])
            values = []
            comment_list.append(lines[file_pos + 1].strip())
            for i in range(num):
                xyz_list = lines[file_pos + i + 2].strip().split(delimiter)
                xyz_list = [x.strip() for x in xyz_list if x != '']
                atom_list = [str(xyz_list[0])]
                xyz_list = [float(x) for x in xyz_list[1:]]
                values.append(atom_list + xyz_list)
            mol_list.append(values)
            file_pos += num + 2
        else:
            # Skip lines
            file_pos += 1
    # close file
    infile.close()
    return mol_list


def write_mol_block_list_to_sdf(mol_block_list, filepath):
    with open(filepath, "w+") as file:
        for mol_block in mol_block_list:
            if mol_block is not None:
                file.write(mol_block)
                file.write("$$$$\n")
            else:
                file.write("".join(["\n",
                                    "     RDKit          2D\n",
                                    "\n",
                                    "  0  0  0  0  0  0  0  0  0  0 V2000\n",
                                    "M  END\n"]))
                file.write("$$$$\n")


def dummy_load_sdf_file(filepath):
    with open(filepath, "r") as f:
        all_sting = f.read()
    return all_sting.split("$$$$\n")