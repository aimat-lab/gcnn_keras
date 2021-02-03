import pickle
import tarfile
import os
import requests
import numpy as np
import shutil
from kgcnn.data.qm.methods import coordinates_to_distancematrix,invert_distance,get_connectivity_from_inversedistancematrix,distance_to_gaussdistance,define_adjacency_from_distance


def qm9_download_dataset(path,overwrite=False):
    """
    Download qm9 dataset as zip-file.
    
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    
    Returns:
        os.path: Filepath
    """
    if(os.path.exists(os.path.join(path,'dsgdb9nsd.xyz.tar.bz2')) == False or overwrite == True):
        print("Downloading dataset...", end='', flush=True)
        data_url = "https://ndownloader.figshare.com/files/3195389"
        r = requests.get(data_url) 
        open(os.path.join(path,'dsgdb9nsd.xyz.tar.bz2'),'wb').write(r.content) 
        print("done")
    else:
        print("Dataset found ... done")
    return os.path.join(path,'dsgdb9nsd.xyz.tar.bz2') 


def qm9_extract_dataset(path,load=False):
    """
    Extract dsgdb9nsd.xyz zip-file.
    
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    
    Returns:
        os.path: Filepath
    """
    if(os.path.exists(os.path.join(path,'dsgdb9nsd.xyz')) == False):
        print("Creating directory ... ", end='', flush=True)
        os.mkdir(os.path.join(path,'dsgdb9nsd.xyz'))
        print("done")
    else:
        print("Directory for extraction exists ... done")
        if(load==False):
            print("Not extracting Zip File ... stopped")
            return os.path.join(path,'dsgdb9nsd.xyz')
    
    print("Read Zip File ... ", end='', flush=True)
    archive = tarfile.open(os.path.join(path,'dsgdb9nsd.xyz.tar.bz2'), "r")
    #Filelistnames = archive.getnames()
    print("done")
    
    print("Extracting Zip folder...", end='', flush=True)
    archive.extractall(os.path.join(path,'dsgdb9nsd.xyz'))
    print("done")
    archive.close()
    
    return os.path.join(path,'dsgdb9nsd.xyz')


def qm9_remove_extracted_dataset(path):
    """
    Remove qm9 extracted folder.

    Args:
        path (str): Parent directory of dsgdb9nsd.xyz.

    Returns:
        None.
    """
    if(os.path.exists(os.path.join(path,'dsgdb9nsd.xyz')) == True):
        print("Clean up unzipped folder...", end='', flush=True)
        shutil.rmtree(os.path.join(path,'dsgdb9nsd.xyz'))
        print("done")
    else:
        print("Cannot find folder dsgdb9nsd.xyz to remove ... Error")


def qm9_write_pickle(path):
    """
    Read .xyz files and store them as pickle python object ca. 200 MB.

    Args:
        path (str): Parent directory with dsgdb9nsd.xyz in it.

    Returns:
        qm9 (list): Full qm9 dataset as python list.
    """
    datasetsize = 133885
    qm9 = []
        
    if(os.path.exists(os.path.join(path,'dsgdb9nsd.xyz')) == False):
        print("Can not find extracted dsgdb9nsd.xyz directory ... Error")
        return qm9
    
    print("Reading dsgdb9nsd files ...", end='', flush=True)
    for i in range(1,datasetsize+1):
        mol = []
        file = "dsgdb9nsd_" + "{:06d}".format(i)+".xyz"
        open_file = open(os.path.join(path,"dsgdb9nsd.xyz",file),"r")
        lines = open_file.readlines()
        mol.append(int(lines[0]))
        labels = lines[1].strip().split(' ')[1].split('\t')
        if(int(labels[0]) != i):
            print("Warning index not matching xyz-file.")
        labels = [int(labels[0])] + [float(x) for x in labels[1:]]
        mol.append(labels)
        cords = []
        for j in range(int(lines[0])):
            atom_info = lines[2+j].strip().split('\t')
            cords.append([atom_info[0]] + [float(x.replace('*^','e')) for x in atom_info[1:]])
        mol.append(cords)
        freqs = lines[int(lines[0]) + 2].strip().split('\t')
        freqs = [float(x) for x in freqs]
        mol.append(freqs)
        smiles = lines[int(lines[0]) + 3].strip().split('\t')
        mol.append(smiles)
        inchis = lines[int(lines[0]) + 4].strip().split('\t')
        mol.append(inchis)
        open_file.close()
        qm9.append(mol)
        #save
    print('done')
    print("Saving qm9.pickle ...", end='', flush=True)
    with open(os.path.join(path,"qm9.pickle"),'wb') as f: 
        pickle.dump(qm9, f)
    print('done')
    return qm9



def make_qm9_graph(qm9,
                   max_distance=4,max_neighbours=15,        
                   gauss_distance = {'GBins' : 20, 'GRange'  : 4, 'GSigma' : 0.4} ):
    """
    Make graph objects from qm9 dataset.

    Args:
        qm9 (list): Full qm9 dataset as python list.
        max_distance (int): 4
        max_neighbours (int): 15
        gauss_distance (dict): {'GBins' : 20, 'GRange'  : 4, 'GSigma' : 0.4}

    Returns:
        list: List of graph props [labels, nodes, edges, edge_idx,gstates]
        
        - labels: All labels of qm9
        - nodes: List of atomic numbers for emebdding layer
        - edges: Edgefeatures (inverse distance, gauss distance)
        - edge_idx: Edge indices (N,2)
        - gstates: Graph states, mean moleculare weight - 7 g/mol
    """
    ## For graph
    
    ## labels
    labels = np.array([x[1] for x in qm9])        
    
    ## Atoms as nodes
    atoms =  [[y[0] for y in x[2]] for x in qm9]
    nodelens = np.array([len(x) for x in atoms],dtype=np.int)
    atom_dict = {'H': 1,'C': 6,'N':7,'O':8,'F':9}
    atom_1hot = {'H': [1,0,0,0,0],'C': [0,1,0,0,0],'N':[0,0,1,0,0],'O':[0,0,0,1,0],'F':[0,0,0,0,1]}
    zval = [[atom_dict[y] for y in x] for x in atoms]
    outzval = [np.array(x,dtype=np.int) for x in zval]
    #outatoms = np.concatenate(outatom,axis=0)
    a1hot = [[atom_1hot[y] for y in x] for x in atoms]
    outa1hot = [np.array(x,dtype=np.float32) for x in a1hot]
    nodes = outzval
    
    ## States
    massdict = {'H': 1.0079,'C': 12.0107,'N': 14.0067,'O': 15.9994,'F': 18.9984}
    mass = [[massdict[y] for y in x] for x in atoms]    
    gstates = np.expand_dims(np.array([np.mean(x) for x in mass]),axis=-1)
    gstates = gstates-7.0 #center at 0
    
    ## Edges
    coord = [[[y[1],y[2],y[3]] for y in x[2]] for x in qm9]
    coord = [np.array(x) for x in coord]
    edge_idx = []
    edges = []
    for i in range(len(labels)):
        xyz = coord[i]
        dist = coordinates_to_distancematrix(xyz)
        invdist = invert_distance(dist)
        ats = outzval[i]
        #cons = get_connectivity_from_inversedistancematrix(invdist,ats)
        cons,_ = define_adjacency_from_distance(dist,max_distance=max_distance,max_neighbours=max_neighbours,exclusive=True,self_loops=False)
        index1 = np.tile(np.expand_dims(np.arange(0,dist.shape[0]),axis=1),(1,dist.shape[1]))
        index2 = np.tile(np.expand_dims(np.arange(0,dist.shape[1]),axis=0),(dist.shape[0],1))
        mask = np.array(cons,dtype=np.bool)
        index12 = np.concatenate([np.expand_dims(index1,axis=-1), np.expand_dims(index2,axis=-1)],axis=-1)
        edge_idx.append(index12[mask])
        if(gauss_distance is not None):
            dist_masked = distance_to_gaussdistance(dist[mask],GBins = gauss_distance['GBins'], GRange = gauss_distance['GRange'], GSigma = gauss_distance['GSigma'])
        else:
            #dist_masked = np.expand_dims(dist[mask],axis=-1)
            dist_masked = np.expand_dims(invdist[mask],axis=-1)
        
        edges.append(dist_masked)
        
    edge_len = np.array([len(x) for x in edge_idx],dtype=np.int)
    #edges = [np.concatenate([edges_inv[i],edges[i]],axis=-1) for i in range(len(edge_idx))]
    edges = [edges[i] for i in range(len(edge_idx))]


    return labels,nodes,edges,edge_idx,gstates



def qm9_graph(max_distance = 4,
        max_neighbours = 15,
        gauss_distance = {'GBins' : 20, 'GRange'  : 4, 'GSigma' : 0.4}
        ):
    """
    Get list of graphs np.arrays for qm9 dataset.
    
    Args:
        max_distance (int): 4
        max_neighbours (int): 15
        gauss_distance (dict): {'GBins' : 20, 'GRange'  : 4, 'GSigma' : 0.4}

    Returns:
        list: List of graph props [labels, nodes, edges, edge_idx, gstates]
        
        - labels: All labels of qm9
        - nodes: List of atomic numbers for emebdding layer
        - edges: Edgefeatures (inverse distance, gauss distance)
        - edge_idx: Edge indices (N,2)
        - gstates: Graph states, mean moleculare weight - 7 g/mol   
    """
    local_path = os.path.split(os.path.realpath(__file__))[0]
    print("Database path:",local_path)
    if(os.path.exists(os.path.join(local_path,"qm9.pickle"))==False):
        qm9_download_dataset(local_path)
        qm9_extract_dataset(local_path)
        qm9 = qm9_write_pickle(local_path)
        qm9_remove_extracted_dataset(local_path)
    else:
        print("Loading qm9.pickle ...", end='', flush=True)
        with open(os.path.join(local_path,"qm9.pickle"),'rb') as f: 
            qm9 = pickle.load(f)
        print('done')
    
    #Make graph
    print("Making graph ...", end='', flush=True)
    out_graph = make_qm9_graph(qm9,max_distance=max_distance,max_neighbours=max_neighbours,gauss_distance=gauss_distance)
    print('done')
    
    return out_graph
        

