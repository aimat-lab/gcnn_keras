import pickle


def save_list(outlist,fname):
    """
    Save a pickled list to file.

    Args:
        outlist (list): List of eg. np.arrays.
        fname (str): Filepath to save.

    Returns:
        None.
    """
    with open(fname,'wb') as f: 
        pickle.dump(outlist, f)


def load_list(fname):
    """
    Load a pickled list from file.

    Args:
        fname (str): Filepath to load.

    Returns:
        outlist (list): Pickle object.
    """
    with open(fname,'rb') as f: 
        outlist = pickle.load(f)
    return outlist



