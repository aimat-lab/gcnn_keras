"""@package: Classes for QM datasets file
Used as a loader.
@author: Patrick
"""

import pandas as pd
import numpy as np
import os
from scipy.io import loadmat

    

class QM7bFile:
    """Class for QM7b Datafile (which is the full dataset). A Fileloader."""

    def __init__(self,filepath=None):
        """Initialize with filepath."""
        #General
        self.DataSetLength = 7211
        self.Zlist = ['','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca']
        
        self.coodinates = None
        self.numatoms = None
        self.ylabels = None
        self.coulmat = None
        self.distmat = None
        self.invdistmat = None
        self.atomlables = None
        self.proton = None
        
        #bond-like properties
        self.bonds_index = None
        self.bonds_coulomb = None
        self.bonds_invdist = None
        self.bonds_dist = None
        
        self.filepath = filepath
        if(filepath != None):
            self.loadQM7b(filepath)
    
    def _coordinates_from_distancematrix(self,DistIn,use_center = None,dim=3):
        """Compute list of coordinates from a distance matrix of shape (N,N)."""
        DistIn = np.array(DistIn)
        dimIn = DistIn.shape[-1]   
        if use_center is None:
            #Take Center of mass (slightly changed for vectorization assuming d_ii = 0)
            di2 = np.square(DistIn)
            di02 = 1/2/dimIn/dimIn*(2*dimIn*np.sum(di2,axis=-1)-np.sum(np.sum(di2,axis=-1),axis=-1))
            MatM = (np.expand_dims(di02,axis=-2) + np.expand_dims(di02,axis=-1) - di2)/2 #broadcasting
        else:
            di2 = np.square(DistIn)
            MatM = (np.expand_dims(di2[...,use_center],axis=-2) + np.expand_dims(di2[...,use_center],axis=-1) - di2 )/2
        u,s,v = np.linalg.svd(MatM)
        vecs = np.matmul(u,np.sqrt(np.diag(s))) # EV are sorted by default
        distout = vecs[...,0:dim]
        return distout
    
    def _coulombmat_to_dist_z(self,coulmat):
        """Cast coulomatrix to distance+atomic number (...,N,N)-> (...,N,N)+(...,N), (...,1)."""
        indslie = np.arange(0,coulmat.shape[-1])
        z = coulmat[...,indslie,indslie]
        prot = np.power(2*z,1/2.4)
        numat = np.sum(prot>0,axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.true_divide(1,prot)
            z[z == np.inf] = 0
            z = np.nan_to_num(z)
        a = np.expand_dims(z, axis = len(z.shape)-1)
        b = np.expand_dims(z, axis = len(z.shape))
        zz = a*b
        c = coulmat*zz
        dinv = np.array(c)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(1,c)
            c[c == np.inf] = 0
            c = np.nan_to_num(c)
        c[...,indslie,indslie] = 0
        dinv[...,indslie,indslie] = 0
        return c,dinv,np.around(prot),numat

    def make_bonds(self,coulomb,dist,invdist):
        """Make bond list of coulomb interactions for (N,N) -> (N*N-N,2),(N*N-N,)."""
        index1 = np.tile(np.expand_dims(np.arange(0,coulomb.shape[0]),axis=1),(1,coulomb.shape[1]))
        index2 = np.tile(np.expand_dims(np.arange(0,coulomb.shape[1]),axis=0),(coulomb.shape[0],1))
        mask = index1 != index2
        index12 = np.concatenate([np.expand_dims(index1,axis=-1), np.expand_dims(index2,axis=-1)],axis=-1)
        return index12[mask],coulomb[mask],dist[mask],invdist[mask]
    
    def loadQM7b(self,filepath=None):
        """Load QM7b dataset from file."""
        if(filepath == None):
            filepath = self.filepath
        dataset = loadmat(filepath)
        y = dataset['T']
        X = dataset['X']
        self.DataSetLength = len(y)
        self.ylabels = y
        
        c,dinv,z,n = self._coulombmat_to_dist_z(X)
        
        self.coulmat = [X[i][:n[i],:n[i]] for i in range(self.DataSetLength)]
        self.proton = [z[i][:n[i]] for i in range(self.DataSetLength)]
        self.invdistmat = [dinv[i][:n[i],:n[i]]*0.52917721090380 for i in range(self.DataSetLength)]
        self.numatoms = n
        self.distmat = [c[i][:n[i],:n[i]]*0.52917721090380 for i in range(self.DataSetLength)]
        self.coodinates =  [self._coordinates_from_distancematrix(x) for x in self.distmat ]
        self.atomlables = [[self.Zlist[int(x)] for x in self.proton[i]] for i in range(self.DataSetLength)]
        
        bonds = [self.make_bonds(self.coulmat[i],self.distmat[i],self.invdistmat[i]) for i in range(self.DataSetLength)]
        
        self.bond_index = [x[0] for x in bonds]
        self.bonds_coulomb = [x[1] for x in bonds]
        self.bonds_dist = [x[2] for x in bonds]
        self.bonds_invdist = [x[3] for x in bonds]
        
        return X,y

