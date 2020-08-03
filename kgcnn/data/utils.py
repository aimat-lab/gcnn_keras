"""@package: utils
@author: Patrick
"""

import numpy as np
import pickle


    
def padd_nested_list(samplelist):
    """recursively padd nested list of numbers or np.arrays to np.array + mask"""
    maxshape = []
    masklist = []
    outlist = []
    #Search for maximum shape with offset this axis
    for i,x in enumerate(samplelist):
        if(isinstance( x,list) == True):
            #If list then recursively go to next depth
            outx,mx = self._padd_list2(x)
            outlist.append(outx)
            masklist.append(mx)
            maxshape.append(outx.shape)
        else:
            #Collect shape and array+mask
            outlist.append(np.array(x))
            masklist.append(np.ones_like(np.array(x),dtype=np.bool))
            maxshape.append(np.array(x).shape)

    #Make full array with maximum shape and zero init
    maxs = np.max(np.array(maxshape),axis=0)
    out = np.zeros([len(samplelist)]+list(maxs))
    mask = np.zeros([len(samplelist)]+list(maxs),dtype=np.bool)
    
    #Fill slices with padded full arrays+mask
    for j in range(len(outlist)):
        its = outlist[j].shape
        sl = tuple([j]+[slice(0,i,1) for i in its])
        out[sl] = outlist[j]
        mask[sl] = masklist[j]
        
    return out,mask


def save(outlist,fname):
    with open(fname,'wb') as f: 
        pickle.dump(outlist, f)

def load(fname):
    with open(fname,'rb') as f: 
        outlist = pickle.load(f)
    return outlist


    



