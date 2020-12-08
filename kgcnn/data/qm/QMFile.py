"""@package: QM dataset
A rudimental class to load the QM7 dataset. Not a proper molecular graph generator.
Used for some examples.

@author: Patrick
"""

import pandas as pd
import numpy as np
import os
from scipy.io import loadmat


class QM7bFile:
    """Class for QM7b Datafile. A Fileloader."""

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
        
        #bond-like properties for all NxN
        self.bonds_index = None
        self.bonds_coulomb = None
        self.bonds_invdist = None
        self.bonds_dist = None
        
        #Connectiviy
        self.adjacency = None
        
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
        """Cast coulomatrix to distance plus atomic number."""
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


    def _get_connectivity_from_inversedistancematrix(self,invdistmat,protons, k1=16.0, k2=4.0/3.0, cutoff=0.85,force_bonds=True): 
        """Find connectivity."""
        #Dictionary of bond radii     
        radii_dict = np.array([0, 0.34, 0.46, 1.2, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.4, 1.25, 1.13, 1.04, 1.1, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.1, 1.07, 1.04, 1.0, 0.99, 1.01, 1.09, 1.12, 1.09, 1.15, 1.1, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.19, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.5, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.3, 1.3, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52, 1.53, 1.54, 1.55])
        #Get Radii 
        protons = np.array(protons,dtype=np.int)
        radii = radii_dict[protons]
        # Calculate
        shape_rad = radii.shape
        r1 = np.expand_dims(radii, axis = len(shape_rad)-1)
        r2 = np.expand_dims(radii, axis = len(shape_rad))
        rmat = r1+r2
        rmat = k2*rmat
        rr = rmat*invdistmat
        damp = (1.0+np.exp(-k1*(rr-1.0)))
        damp = 1.0/damp 
        reorder_axis = np.arange(len(invdistmat.shape))
        reorder_axis[-1] = np.arange(len(invdistmat.shape))[-2]
        reorder_axis[-2] =  np.arange(len(invdistmat.shape))[-1]
        if(force_bonds==True): # Have at least one bond
            damp_sym = ( damp + np.transpose(damp,reorder_axis))/2
            maxvals = np.expand_dims(np.argmax(damp,axis=-1),axis=-1)
            #damp[...,np.arange(0,damp.shape[-2]),maxvals]=1
            np.put_along_axis(damp,maxvals,1,axis=-1)
            damp = np.transpose(damp,reorder_axis)
            np.put_along_axis(damp,maxvals,1,axis=-1)
        damp[damp<cutoff] = 0
        bond_tab = np.array(np.round(damp),dtype = np.bool)
        return bond_tab

    def _make_bonds(self,proton,coulomb,dist,invdist):
        """Make bond list of coulomb interactions for (N,N) -> (N*N-N,2),(N*N-N,)."""
        index1 = np.tile(np.expand_dims(np.arange(0,coulomb.shape[0]),axis=1),(1,coulomb.shape[1]))
        index2 = np.tile(np.expand_dims(np.arange(0,coulomb.shape[1]),axis=0),(coulomb.shape[0],1))
        mask = np.array(self._get_connectivity_from_inversedistancematrix(invdist,proton),dtype=np.bool)
        index12 = np.concatenate([np.expand_dims(index1,axis=-1), np.expand_dims(index2,axis=-1)],axis=-1)
        return index12[mask],coulomb[mask],dist[mask],invdist[mask],mask
    
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
        self.invdistmat = [dinv[i][:n[i],:n[i]]/0.52917721090380 for i in range(self.DataSetLength)]
        self.numatoms = n
        self.distmat = [c[i][:n[i],:n[i]]*0.52917721090380 for i in range(self.DataSetLength)]
        self.coodinates =  [self._coordinates_from_distancematrix(x) for x in self.distmat ]
        self.atomlables = [[self.Zlist[int(x)] for x in self.proton[i]] for i in range(self.DataSetLength)]
                
        bonds = [self._make_bonds(self.proton[i],self.coulmat[i],self.distmat[i],self.invdistmat[i]) for i in range(self.DataSetLength)]
        
        self.bond_index = [x[0] for x in bonds]
        self.bonds_coulomb = [x[1] for x in bonds]
        self.bonds_dist = [x[2] for x in bonds]
        self.bonds_invdist = [x[3] for x in bonds]
        self.adjacency = [x[4] for x in bonds]
        
        return X,y

