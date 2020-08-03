"""@package: Classes for QM datasets file
Used as a loader.
@author: Patrick
"""

import pandas as pd
import numpy as np
import os
from scipy.io import loadmat


class QM9File:
    """
    Class to represent a QM9 Datafile. Used as a Fileloader. 
    """
    #static
    AtomDictionary = {'H': 1,'C': 6,'N':7,'O':8,'F':9}
    LabelsProperty = ["index","A","B","C","mu","alpha","homo","lumo","gap","r2","zpve","U0","U","H","G","Cv"]
    
    def __init__(self,filepath=None):
        
        #Paramter Molecule 
        self.NumAtom = None
        self.ProtonNumber = None
        self.AtomList = None
        self.Coord3DAtom = None
        self.NumFreq = None
        self.Freq = None
        self.Mulligan = None
        self.Smiles = None
        self.InChI = None
        self.Labels = None
        
        self.filepath = filepath  
        if(filepath != None):
            self.loadQM9XYZ(filepath)
        
    def loadQM9XYZ(self,filepath):
        """
        loading function of Qm9File
        @TODO: only use read_csv once for total file and then read values
        """
        self.NumAtom = int(pd.read_csv(filepath,engine='python',nrows=1,header=None).loc[0,0])
        self.Labels = pd.read_csv(filepath,sep=' |\t',engine='python',skiprows=1,nrows=1,header=None).loc[0,1:].values.tolist()
        tempAtomCoord = pd.read_csv(filepath,sep='\t',engine='python', skiprows=2, skipfooter=3,header=None)
        for j in range(1,5):
            if tempAtomCoord[j].dtype == 'O':
                tempAtomCoord[j] = tempAtomCoord[j].str.replace('*^','e',regex=False).astype(float)
        self.AtomList = tempAtomCoord.loc[:,0].tolist()
        self.Coord3DAtom = np.array(tempAtomCoord.loc[:,1:3])
        self.Mulligan = np.array(tempAtomCoord.loc[:,4])
        tempFreq = pd.read_csv(filepath,sep=' |\t',engine='python',skiprows=self.NumAtom+2,nrows=1,header=None)
        self.NumFreq = int(tempFreq.shape[1])
        self.Freq = np.array(tempFreq)
        self.Smiles = pd.read_csv(filepath,sep=' |\t',engine='python',skiprows=self.NumAtom+3,nrows=1,header=None).iloc[0,:].tolist()
        self.InChI = pd.read_csv(filepath,sep=' |\t',engine='python',skiprows=self.NumAtom+4,nrows=1,header=None).iloc[0,:].tolist()
        #Calculate ProtonNumber
        self.ProtonNumber = pd.DataFrame(self.AtomList).replace(self.AtomDictionary)[0].tolist()
        self.ProtonNumber = np.array(self.ProtonNumber)
    

class QM7bFile:
    """
    Class for QM7b Datafile (which is the full database). A Fileloader. 
    """
    def __init__(self,filepath=None):
        #General
        self.DataSetLength = 7211
        self.Zlist = ['','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca']
        
        self.coodinates = None
        self.numatoms = None
        self.ylabels = None
        self.coulmat = None
        self.distmat = None
        self.atomlables = None
        self.proton = None
        
        self.bonds = None
        self.bondcoulomb = None
        
        self.filepath = filepath
        if(filepath != None):
            self.loadQM7b(filepath)
    
    def _coordinates_from_distancematrix(self,DistIn,use_center = None,dim=3):
        """Computes list of coordinates from a distance matrix of shape (N,N)"""
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
        """ coulomatrix to distance+atomic number (...,N,N)-> (...,N,N)+(...,N), (...,1)"""
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
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(1,c)
            c[c == np.inf] = 0
            c = np.nan_to_num(c)
        c[...,indslie,indslie] = 0
        return c,np.around(prot),numat

    
    def _bonds(self,coulomb):
        """bond list of coulomb interactions for (N,N) -> (N*N-N,2),(N*N-N,)"""
        index1 = np.tile(np.expand_dims(np.arange(0,coulomb.shape[0]),axis=1),(1,coulomb.shape[1]))
        index2 = np.tile(np.expand_dims(np.arange(0,coulomb.shape[1]),axis=0),(coulomb.shape[0],1))
        mask = index1 != index2
        index12 = np.concatenate([np.expand_dims(index1,axis=-1), np.expand_dims(index2,axis=-1)],axis=-1)
        return index12[mask],coulomb[mask]
    
    def loadQM7b(self,filepath=None):
        """Load QM7b dataset from file."""
        if(filepath == None):
            filepath = self.filepath
        dataset = loadmat(filepath)
        y = dataset['T']
        X = dataset['X']
        self.DataSetLength = len(y)
        self.ylabels = y
        
        c,z,n = self._coulombmat_to_dist_z(X)
        self.coulmat = [X[i][:n[i],:n[i]] for i in range(self.DataSetLength)]
        self.proton = [z[i][:n[i]] for i in range(self.DataSetLength)]
        self.numatoms = n
        
        dlist = []
        zlist = []
        atstr = []
        for i in range(0,self.DataSetLength):
            dlist.append(c[i][:n[i],:n[i]]*0.52917721090380)#bohr to A
            z_i = z[i][:n[i]]
            zlist.append(z_i)
            atstr.append([self.Zlist[int(x)] for x in z_i])
        
        clist =  [self._coordinates_from_distancematrix(x) for x in dlist]
        self.coodinates = clist
        self.atomlables = atstr
        self.distmat = dlist
        
        bonds = [self._bonds(x) for x in self.coulmat]
        self.bonds = [x[0] for x in bonds]
        self.bondcoulomb = [x[1] for x in bonds]
        
        return X,y

