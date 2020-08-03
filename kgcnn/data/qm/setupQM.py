"""@package: Download database
The qm7b dataset and corresponding features can be easily loaded into memory.
@author: Patrick Reiser
"""

import tarfile
import os
import requests
import numpy as np
import pandas as pd
 


def qm7b_download_dataset(path,overwrite=False):
    """
    Downloads the Qm7b dataset from http://www.quantum-machine.org/data/qm7b.mat
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    """
    datapath = os.path.join(path,'qm7b.mat')
    if(os.path.exists(datapath) == False or overwrite == True):
        print("downloading dataset...",end='', flush=True)
        data_url = "http://www.quantum-machine.org/data/qm7b.mat"
        r = requests.get(data_url) 
        open(datapath,'wb').write(r.content) 
        print("done")
    else:
        print("QM7b database already exists")
        
    return datapath


def qm7_download_dataset(path,overwrite=False):
    """
    Downloads the Qm7 dataset from http://www.quantum-machine.org/data/qm7.mat
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    """
    datapath = os.path.join(path,'qm7.mat')
    if(os.path.exists(datapath) == False or overwrite == True):
        print("downloading dataset...",end='', flush=True)
        data_url = "http://quantum-machine.org/data/qm7.mat"
        r = requests.get(data_url) 
        open(datapath,'wb').write(r.content) 
        print("done")
    else:
        print("QM7 database already exists")
        
    return datapath



def qm9_download_dataset(path,overwrite=False):
    """
    Downloads qm9 dataset as zip-file
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    """
    if(os.path.exists(os.path.join(path,'dsgdb9nsd.xyz.tar.bz2')) == False or overwrite == True):
        print("downloading dataset...", end='', flush=True)
        data_url = "https://ndownloader.figshare.com/files/3195389"
        r = requests.get(data_url) 
        open(os.path.join(path,'dsgdb9nsd.xyz.tar.bz2'),'wb').write(r.content) 
        print("done")
    else:
        print("Dataset found ... done")
    return os.path.join(path,'dsgdb9nsd.xyz.tar.bz2') 


def qm9_extract_dataset(path,load=False):
    """
    Extracts dsgdb9nsd.xyz zip-file and puts out a FileList
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    """
    if(os.path.exists(os.path.join(path,'dsgdb9nsd.xyz')) == False):
        print("creating directory ... ", end='', flush=True)
        os.mkdir(os.path.join(path,'dsgdb9nsd.xyz'))
        print("done")
    else:
        print("Directory for extraction exists")
        if(load==False):
            print("not extracting Zip File...stopped")
            return os.path.join(path,'dsgdb9nsd.xyz')
    
    print("Read Zip File ... ", end='', flush=True)
    archive = tarfile.open(os.path.join(path,'dsgdb9nsd.xyz.tar.bz2'), "r")
    Filelistnames = archive.getnames()
    print("done")
    
    print("Extracting Zip folder...", end='', flush=True)
    archive.extractall(os.path.join(path,'dsgdb9nsd.xyz'))
    print("done")
    
    print("Saving Filelist ...", end='', flush=True)
    QM9FilelistPD = pd.DataFrame(Filelistnames)
    QM9FilelistPD.to_csv(os.path.join(path,"QM9FileList"),index=True,sep='\t',header=False)
    print("done")
    
    return os.path.join(path,'dsgdb9nsd.xyz')



