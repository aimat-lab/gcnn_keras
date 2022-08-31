# Summary of Benchmark Training

Note that these are the results for models within `kgcnn` implementation, and that training is not always done with optimal hyperparameter or splits, when comparing with literature.
This table is generated automatically from keras history logs.
Model weights and training statistics plots are not uploaded on github due to their file size.

## CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse node attributes and 7 node classes.

| model | kgcnn | epochs | Categorical accuracy | 
| :---: | :---: | :---: | :---: | 
| GAT | 2.1.0 | 250 | 0.8645 &pm; 0.0129  | 
| GATv2 | 2.1.0 | 250 | 0.8349 &pm; 0.0150  | 
| GCN | 2.1.0 | 300 | 0.7969 &pm; 0.0144  | 
| GIN | 2.1.0 | 500 | 0.9477 &pm; 0.0138  | 
| GraphSAGE | 2.1.0 | 500 | 0.9694 &pm; 0.0026  | 

## CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes.

| model | kgcnn | epochs | Categorical accuracy | 
| :---: | :---: | :---: | :---: | 
| GCN | 2.1.0 | 300 | 0.6150 &pm; 0.0121  | 

## ESOLDataset

ESOL (MoleculeNet) consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). Here we use a random split.

| model | kgcnn | epochs | MAE [log mol/L] | RMSE [log mol/L] | 
| :---: | :---: | :---: | :---: | :---: | 
| AttentiveFP | 2.1.0 | 200 | 0.4359 &pm; 0.0295  | 0.5920 &pm; 0.0307  | 
| CMPNN | 2.1.0 | 600 | 0.4740 &pm; 0.0259  | 0.6766 &pm; 0.0266  | 
| DimeNetPP | 2.1.0 | 872 | 0.4572 &pm; 0.0304  | 0.6377 &pm; 0.0501  | 
| DMPNN | 2.1.0 | 300 | 0.4381 &pm; 0.0203  | 0.6321 &pm; 0.0478  | 
| GAT | 2.1.0 | 500 | 0.4699 &pm; 0.0435  | 0.6711 &pm; 0.0745  | 
| GATv2 | 2.1.0 | 500 | 0.4628 &pm; 0.0432  | 0.6615 &pm; 0.0565  | 
| GCN | 2.1.0 | 800 | 0.5639 &pm; 0.0102  | 0.7995 &pm; 0.0324  | 
| GIN | 2.1.0 | 300 | 0.5107 &pm; 0.0395  | 0.7241 &pm; 0.0441  | 
| GIN.make_model_edge | 2.1.0 | 300 | 0.4761 &pm; 0.0259  | 0.6733 &pm; 0.0407  | 
| GraphSAGE | 2.1.0 | 500 | 0.4654 &pm; 0.0377  | 0.6556 &pm; 0.0697  | 
| HamNet | 2.1.0 | 400 | 0.5492 &pm; 0.0509  | 0.7645 &pm; 0.0676  | 
| INorp | 2.1.0 | 500 | 0.4828 &pm; 0.0201  | 0.6748 &pm; 0.0350  | 
| Megnet | 2.1.0 | 800 | 0.5597 &pm; 0.0314  | 0.7972 &pm; 0.0439  | 
| NMPN | 2.1.0 | 800 | 0.5706 &pm; 0.0497  | 0.8144 &pm; 0.0710  | 
| PAiNN | 2.1.0 | 250 | 0.4182 &pm; 0.0198  | 0.5961 &pm; 0.0344  | 
| Schnet | 2.1.0 | 800 | 0.4682 &pm; 0.0272  | 0.6539 &pm; 0.0471  | 

## LipopDataset

Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles and their corresponding octanol/water distribution coefficient (logD at pH 7.4). Here we use a random split.

| model | kgcnn | epochs | MAE [log mol/L] | RMSE [log mol/L] | 
| :---: | :---: | :---: | :---: | :---: | 
| AttentiveFP | 2.1.0 | 200 | 0.4644 &pm; 0.0245  | 0.6393 &pm; 0.0408  | 
| DMPNN | 2.1.0 | 300 | 0.3781 &pm; 0.0091  | 0.5440 &pm; 0.0162  | 
| GAT | 2.1.0 | 500 | 0.5034 &pm; 0.0060  | 0.7037 &pm; 0.0202  | 
| GATv2 | 2.1.0 | 500 | 0.3971 &pm; 0.0238  | 0.5688 &pm; 0.0609  | 
| GIN | 2.1.0 | 300 | 0.4503 &pm; 0.0106  | 0.6175 &pm; 0.0210  | 
| HamNet | 2.1.0 | 400 | 0.4535 &pm; 0.0119  | 0.6305 &pm; 0.0244  | 
| INorp | 2.1.0 | 500 | 0.4668 &pm; 0.0118  | 0.6576 &pm; 0.0214  | 
| PAiNN | 2.1.0 | 250 | 0.4050 &pm; 0.0070  | 0.5837 &pm; 0.0162  | 

## MatProjectEFormDataset

Materials Project dataset from Matbench with 132752 crystal structures and their corresponding formation energy in [eV/atom].

| model | kgcnn | epochs | MAE [eV/atom] | RMSE [eV/atom] | 
| :---: | :---: | :---: | :---: | :---: | 
| Schnet.make_crystal_model | 2.1.0 | 800 | 0.0209 &pm; 0.0004  | 0.0514 &pm; 0.0028  | 

