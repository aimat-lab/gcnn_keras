"""@package: utils
@author: Patrick, 
"""

import numpy as np
import pickle


def node_list_to_ragged(nodelist):
    pass

def edge_list_to_ragged(nodelist):
    pass

def save_list(outlist,fname):
    with open(fname,'wb') as f: 
        pickle.dump(outlist, f)

def load_list(fname):
    with open(fname,'rb') as f: 
        outlist = pickle.load(f)
    return outlist


    



