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


def convert_xyz_to_mol_ob(xyz_str: str):
    """Conversion of xyz-string to mol-string.

    The order of atoms in the list should be the same as output. Uses openbabel for conversion.

    Args:
        xyzs (str): Convert the xyz string to mol-string
    Returns:
        str: Mol-string from xyz-information. Generates structure or bond information.
    """
    try:
        from openbabel import openbabel
    except ImportError:
        raise ImportError("ERROR:kgcnn: Conversion from xyz to mol requires openbabel. Please install openbabel")

    ob_conversion = openbabel.OBConversion()
    ob_conversion.SetInAndOutFormats("xyz", "mol")
    # ob_conversion.SetInFormat("xyz")

    mol = openbabel.OBMol()
    ob_conversion.ReadString(mol, xyz_str)
    # print(xyz_str)

    out_mol = ob_conversion.WriteString(mol)
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
        for a in lines[4:(na+4)]:
            atoms.append([float(x) for x in a[:3]] + a[3:])
        out_list.append(atoms)
        # bond block
        bonds = []
        for b in lines[4+na:4+na+nb]:
            bonds.append([int(x) for x in b])
        out_list.append(bonds)
        # Properties block
        prop_block = []
        for p in lines[4+na+nb:]:
            if len(p) >= 2:
                if p[0] == "M" and p[1] != "END":
                    prop_block.append(p)
        out_list.append(prop_block)
    else:
        raise NotImplementedError("ERROR:kgcnn: Can not parse mol V3000 or higher.")
    return out_list