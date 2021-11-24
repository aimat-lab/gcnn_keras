try:
    from openbabel import openbabel
except ImportError:
    print("ERROR:kgcnn: Conversion from xyz to mol requires openbabel. Please install openbabel")


def convert_xyz_to_mol_ob(xyz_str: str, stop_logging: bool = True):
    """Conversion of xyz-string to mol-string.

    The order of atoms in the list should be the same as output. Uses openbabel for conversion.

    Args:
        xyz_str (str): Convert the xyz string to mol-string
        stop_logging (bool): Whether to stop logging. Default is True.
    Returns:
        str: Mol-string from xyz-information. Generates structure or bond information.
    """
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