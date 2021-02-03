import pickle
import zipfile
import os
import requests
import numpy as np
import shutil



def mutagenicity_download_dataset(path,overwrite=False):
    """
    Download Mutagenicity as zip-file.
    
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    
    Returns:
        os.path: Filepath
    """
    if(os.path.exists(os.path.join(path,'Mutagenicity.zip')) == False or overwrite == True):
        print("Downloading dataset...", end='', flush=True)
        data_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/Mutagenicity.zip"
        r = requests.get(data_url) 
        open(os.path.join(path,'Mutagenicity.zip'),'wb').write(r.content) 
        print("done")
    else:
        print("Dataset found ... done")
    return os.path.join(path,'Mutagenicity.zip') 


def mutagenicity_extract_dataset(path,load=False):
    """
    Extract Mutagenicity.zip zip-file.
    
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    
    Returns:
        os.path: Filepath
    """
    if(os.path.exists(os.path.join(path,'Mutagenicity')) == True):
        print("Directory for extraction exists ... done")
        if(load==False):
            print("Not extracting Zip File ... stopped")
            return os.path.join(path,'Mutagenicity')
    
    print("Read Zip File ... ", end='', flush=True)
    archive =  zipfile.ZipFile(os.path.join(path,'Mutagenicity.zip'), "r")
    #Filelistnames = archive.getnames()
    print("done")
    
    print("Extracting Zip folder...", end='', flush=True)
    archive.extractall(os.path.join(path))
    print("done")
    archive.close()
    
    return os.path.join(path,'Mutagenicity')


def mutagenicity_load(path):
    """
    Load Mutagenicity dataset as list.

    Args:
        path (str): path location.

    Returns:
        list: [labels, nodes, edge_indices, edges, atoms]
        
        - labels (list): Mutagenity label (0,1).
        - nodes (list): Atoms as Atomnumber array.
        - edge_indices (list): Bond indices (i,j).
        - edges (list): Bond type.
        - atoms (list): Atom list as string.
    """
    # path = os.path.split(os.path.realpath(__file__))[0]
    mutagenicity_download_dataset(path)
    mutagenicity_extract_dataset(path)
    ##A
    mutag_A = []
    open_file = open(os.path.join(path,"Mutagenicity","Mutagenicity_A.txt"),"r")
    for lines in open_file.readlines():
        idxs = lines.strip().split(',')
        idxs = [int(x) for x in idxs]
        mutag_A.append(idxs)
    open_file.close()
    mutag_A = np.array(mutag_A)
    ##edge_labels
    mutag_e = []
    open_file = open(os.path.join(path,"Mutagenicity","Mutagenicity_edge_labels.txt"),"r")
    for lines in open_file.readlines():
        idxs = int(lines.strip())
        mutag_e.append(idxs)
    open_file.close()
    ##graph indicator
    mutag_gi = []
    open_file = open(os.path.join(path,"Mutagenicity","Mutagenicity_graph_indicator.txt"),"r")
    for lines in open_file.readlines():
        idxs = int(lines.strip())
        mutag_gi.append(idxs)
    open_file.close()
    ##graph labels
    mutag_gl = []
    open_file = open(os.path.join(path,"Mutagenicity","Mutagenicity_graph_labels.txt"),"r")
    for lines in open_file.readlines():
        idxs = int(lines.strip())
        mutag_gl.append(idxs)
    open_file.close()
    ##node labels
    mutag_n = []
    open_file = open(os.path.join(path,"Mutagenicity","Mutagenicity_node_labels.txt"),"r")
    for lines in open_file.readlines():
        idxs = int(lines.strip())
        mutag_n.append(idxs)
    open_file.close()
    
    #cast to numpy
    mutag_A = np.array(mutag_A,dtype=np.int)
    mutag_e = np.array(mutag_e,dtype=np.int)
    mutag_gi = np.array(mutag_gi,dtype=np.int)
    mutag_gl = np.array(mutag_gl,dtype=np.int)
    mutag_n = np.array(mutag_n,dtype=np.int)
    
    #labels
    labels = np.array(mutag_gl,dtype=np.int)
    N_data = len(labels)
    
    #shift index
    mutag_A = mutag_A-1
    mutag_gi = mutag_gi-1
    
    #split into sperate graphs
    graph_id, counts = np.unique(mutag_gi, return_counts=True)
    graphlen = np.zeros(N_data ,dtype=np.int)
    graphlen[graph_id] = counts
    nodes0123 = np.split(mutag_n, np.cumsum(graphlen)[:-1])
    node_translate = np.array([6,8,17,1,7,9,35,16,15,53,11,19,3,20],dtype=np.int)
    atoms_translate = ['C','O','Cl','H','N','F','Br','S','P','I','Na','K','Li','Ca']
    z_translate = {node_translate[i] : atoms_translate[i]  for i in range(len(node_translate))}
    nodes = [node_translate[x] for x in nodes0123]
    atoms = [[atoms_translate[y] for y in x] for x in nodes0123]
    
    #edge_indicator
    graph_id_edge = mutag_gi[mutag_A[:,0]] #is the same for A[:,1]
    graph_id2, counts_edge = np.unique(graph_id_edge, return_counts=True)
    edgelen = np.zeros(N_data ,dtype=np.int)
    edgelen[graph_id2] = counts_edge
    edges = np.split(mutag_e+1, np.cumsum(edgelen)[:-1])
    
    #indices
    node_index = np.concatenate([np.arange(x) for x in graphlen],axis=0)
    edge_indices = node_index[mutag_A]
    edge_indices = np.split(edge_indices, np.cumsum(edgelen)[:-1])
    
    # Require cleaning steps
    labels_clean = []
    nodes_clean = []
    edge_indices_clean = []
    edges_clean = []
    atoms_clean = []
    
    #Remove unconnected atoms. not Na Li etc.
    print("Checking Database...")
    for i in range(len(nodes)):
        nats = nodes[i]
        cons = np.arange(len(nodes[i])) 
        test_cons = np.sort(np.unique(edge_indices[i].flatten()))
        is_cons = np.zeros_like(cons,dtype=np.bool)
        is_cons[test_cons] = True
        is_cons[nats == 20] = True 
        is_cons[nats == 3] = True
        is_cons[nats == 19] = True
        is_cons[nats == 11] = True
        if(np.sum(is_cons) != len(cons)):
            info_list = nodes[i][is_cons==False]
            info_list,info_cnt = np.unique(info_list,return_counts=True)
            info_list = {z_translate[info_list[j]]:info_cnt[j] for j in range(len(info_list))}
            print("Removing unconnected",info_list,"from molecule",i)
            nodes_clean.append(nats[is_cons])
            atoms_clean.append([atoms[i][j] for j in range(len(is_cons)) if is_cons[j] == True])
            #Need to correct indices
            indices_used = cons[is_cons]
            indices_new = np.arange(len(indices_used))
            indices_old = np.zeros(len(nodes[i]),dtype=np.int) 
            indices_old[indices_used] = indices_new
            edge_idx_new = indices_old[edge_indices[i]]
            edge_indices_clean.append(edge_idx_new)
        else:
            nodes_clean.append(nats)
            atoms_clean.append(atoms[i])
            edge_indices_clean.append(edge_indices[i])
        edges_clean.append(edges[i])
        labels_clean.append(labels[i])
    
    print("Note: Database still has unconnected Na+,Li+,K+ etc.")

    #return labels,nodes,edge_indices,edges,atoms
    return labels_clean,nodes_clean,edge_indices_clean,edges_clean,atoms_clean


def mutagenicity_graph():
    """
    Generate list of mutagenicity graphs.

    Returns:
        ist: [labels, nodes, edge_indices, edges, atoms]
        
        - labels (list): Mutagenity label (0,1).
        - nodes (list): Atoms as Atomnumber array.
        - edge_indices (list): Bond indices (i,j).
        - edges (list): Bond type.
        - atoms (list): Atom list as string.
    """
    local_path = os.path.split(os.path.realpath(__file__))[0]
    print("Database path:",local_path)
    if(os.path.exists(os.path.join(local_path,"Mutagenicity"))==False):
        mutagenicity_download_dataset(local_path)
        mutagenicity_extract_dataset(local_path)
    
    data = mutagenicity_load(local_path)
    return data



# labels,nodes,edge_indices,edges,atoms = mutagenicity_graph()

# import rdkit
# import rdkit.Chem.AllChem
# import numpy as np

# def rdkit_mol_from_atoms_bonds(atoms,bonds,sani=False):
#     bond_names =  {'AROMATIC': rdkit.Chem.rdchem.BondType.AROMATIC, 'DATIVE': rdkit.Chem.rdchem.BondType.DATIVE, 'DATIVEL': rdkit.Chem.rdchem.BondType.DATIVEL, 'DATIVEONE': rdkit.Chem.rdchem.BondType.DATIVEONE, 'DATIVER': rdkit.Chem.rdchem.BondType.DATIVER, 'DOUBLE': rdkit.Chem.rdchem.BondType.DOUBLE, 'FIVEANDAHALF': rdkit.Chem.rdchem.BondType.FIVEANDAHALF, 'FOURANDAHALF': rdkit.Chem.rdchem.BondType.FOURANDAHALF, 'HEXTUPLE': rdkit.Chem.rdchem.BondType.HEXTUPLE, 'HYDROGEN': rdkit.Chem.rdchem.BondType.HYDROGEN, 'IONIC': rdkit.Chem.rdchem.BondType.IONIC, 'ONEANDAHALF': rdkit.Chem.rdchem.BondType.ONEANDAHALF, 'OTHER': rdkit.Chem.rdchem.BondType.OTHER, 'QUADRUPLE': rdkit.Chem.rdchem.BondType.QUADRUPLE, 'QUINTUPLE': rdkit.Chem.rdchem.BondType.QUINTUPLE, 'SINGLE': rdkit.Chem.rdchem.BondType.SINGLE, 'THREEANDAHALF': rdkit.Chem.rdchem.BondType.THREEANDAHALF, 'THREECENTER': rdkit.Chem.rdchem.BondType.THREECENTER, 'TRIPLE': rdkit.Chem.rdchem.BondType.TRIPLE, 'TWOANDAHALF': rdkit.Chem.rdchem.BondType.TWOANDAHALF, 'UNSPECIFIED': rdkit.Chem.rdchem.BondType.UNSPECIFIED, 'ZERO': rdkit.Chem.rdchem.BondType.ZERO}
#     bond_vals = {0: rdkit.Chem.rdchem.BondType.UNSPECIFIED, 1: rdkit.Chem.rdchem.BondType.SINGLE, 2: rdkit.Chem.rdchem.BondType.DOUBLE, 3: rdkit.Chem.rdchem.BondType.TRIPLE, 4: rdkit.Chem.rdchem.BondType.QUADRUPLE, 5: rdkit.Chem.rdchem.BondType.QUINTUPLE, 6: rdkit.Chem.rdchem.BondType.HEXTUPLE, 7: rdkit.Chem.rdchem.BondType.ONEANDAHALF, 8: rdkit.Chem.rdchem.BondType.TWOANDAHALF, 9: rdkit.Chem.rdchem.BondType.THREEANDAHALF, 10: rdkit.Chem.rdchem.BondType.FOURANDAHALF, 11: rdkit.Chem.rdchem.BondType.FIVEANDAHALF, 12: rdkit.Chem.rdchem.BondType.AROMATIC, 13: rdkit.Chem.rdchem.BondType.IONIC, 14: rdkit.Chem.rdchem.BondType.HYDROGEN, 15: rdkit.Chem.rdchem.BondType.THREECENTER, 16: rdkit.Chem.rdchem.BondType.DATIVEONE, 17: rdkit.Chem.rdchem.BondType.DATIVE, 18: rdkit.Chem.rdchem.BondType.DATIVEL, 19: rdkit.Chem.rdchem.BondType.DATIVER, 20: rdkit.Chem.rdchem.BondType.OTHER, 21: rdkit.Chem.rdchem.BondType.ZERO}
    
#     mol = rdkit.Chem.RWMol()
#     for atm in atoms:
#         mol.AddAtom(rdkit.Chem.Atom(atm))
    
#     for i in range(len(bonds)):
#         if(not mol.GetBondBetweenAtoms(int(bonds[i][0]),int(bonds[i][1])) and int(bonds[i][0]) != int(bonds[i][1])):
#             if(len(bonds[i]) == 3):
#                 bi = bonds[i][2]
#                 if(isinstance(bi,str)):
#                     bond_type = bond_names[bi]
#                 elif(isinstance(bi,int)):
#                     bond_type = bond_vals[bi]
#                 else:
#                     bond_type = bi #or directly rdkit.Chem.rdchem.BondType
#                 mol.AddBond(int(bonds[i][0]), int(bonds[i][1]), bond_type)
#             else:
#                 mol.AddBond(int(bonds[i][0]), int(bonds[i][1]))
    
#     mol = mol.GetMol()
    
#     if(sani == True):
#         rdkit.Chem.SanitizeMol(mol)
    
#     return mol

# mol_list = []
# for rd_idx in range(len(nodes)):
#     bonds = np.concatenate([edge_indices[rd_idx],np.expand_dims(edges[rd_idx],axis=-1)],axis=-1).tolist()
#     mol = rdkit_mol_from_atoms_bonds(atoms[rd_idx],bonds)
#     mol_list.append(mol)
