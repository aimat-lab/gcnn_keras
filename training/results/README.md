# Summary of Benchmark Training

Note that these are the results for models within `kgcnn` implementation, and that training is not always done with optimal hyperparameter or splits, when comparing with literature.
This table is generated automatically from keras history logs.
Model weights and training statistics plots are not uploaded on github due to their file size.

## CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse node attributes and 7 node classes. Here we use random 5-fold cross-validation  on nodes.

| model | kgcnn | epochs | Categorical accuracy | 
| :---: | :---: | :---: | :---: | 
| GAT | 2.1.0 | 250 | 0.8667 &pm; 0.0069  | 
| GATv2 | 2.1.0 | 250 | 0.8379 &pm; 0.0158  | 
| GCN | 2.1.0 | 300 | 0.8047 &pm; 0.0113  | 
| GIN | 2.1.0 | 500 | 0.8427 &pm; 0.0165  | 
| GraphSAGE | 2.1.0 | 500 | 0.8486 &pm; 0.0097  | 

## CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes.

| model | kgcnn | epochs | Categorical accuracy | 
| :---: | :---: | :---: | :---: | 
| GAT | 2.1.0 | 250 | 0.6765 &pm; 0.0069  | 
| GATv2 | 2.1.0 | 250 | 0.3320 &pm; 0.0252  | 
| GCN | 2.1.0 | 300 | 0.6156 &pm; 0.0052  | 
| GIN | 2.1.0 | 800 | 0.6368 &pm; 0.0077  | 
| GraphSAGE | 2.1.0 | 600 | 0.6151 &pm; 0.0058  | 

## ESOLDataset

ESOL (MoleculeNet) consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation.

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

Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles and their corresponding octanol/water distribution coefficient (logD at pH 7.4). We use random 5-fold cross-validation.

| model | kgcnn | epochs | MAE [log mol/L] | RMSE [log mol/L] | 
| :---: | :---: | :---: | :---: | :---: | 
| AttentiveFP | 2.1.0 | 200 | 0.4644 &pm; 0.0245  | 0.6393 &pm; 0.0408  | 
| CMPNN | 2.1.0 | 600 | 0.4131 &pm; 0.0061  | 0.5835 &pm; 0.0094  | 
| DMPNN | 2.1.0 | 300 | 0.3781 &pm; 0.0091  | 0.5440 &pm; 0.0162  | 
| GAT | 2.1.0 | 500 | 0.5034 &pm; 0.0060  | 0.7037 &pm; 0.0202  | 
| GATv2 | 2.1.0 | 500 | 0.3971 &pm; 0.0238  | 0.5688 &pm; 0.0609  | 
| GIN | 2.1.0 | 300 | 0.4503 &pm; 0.0106  | 0.6175 &pm; 0.0210  | 
| HamNet | 2.1.0 | 400 | 0.4535 &pm; 0.0119  | 0.6305 &pm; 0.0244  | 
| INorp | 2.1.0 | 500 | 0.4668 &pm; 0.0118  | 0.6576 &pm; 0.0214  | 
| PAiNN | 2.1.0 | 250 | 0.4050 &pm; 0.0070  | 0.5837 &pm; 0.0162  | 
| Schnet | 2.1.0 | 800 | 0.4879 &pm; 0.0205  | 0.6535 &pm; 0.0320  | 

## MatProjectEFormDataset

Materials Project dataset from Matbench with 132752 crystal structures and their corresponding formation energy in [eV/atom]. We use random 10-fold cross-validation.

| model | kgcnn | epochs | MAE [eV/atom] | RMSE [eV/atom] | 
| :---: | :---: | :---: | :---: | :---: | 
| CGCNN.make_crystal_model | 2.1.0 | 1000 | 0.0355 &pm; 0.0005  | 0.0851 &pm; 0.0035  | 
| Megnet.make_crystal_model | 2.1.0 | 1000 | 0.0241 &pm; 0.0006  | 0.0642 &pm; 0.0025  | 
| PAiNN.make_crystal_model | 2.1.0 | 800 | nan &pm; nan  | nan &pm; nan  | 
| Schnet.make_crystal_model | 2.1.0 | 800 | 0.0209 &pm; 0.0004  | 0.0514 &pm; 0.0028  | 

## MutagenicityDataset

Mutagenicity dataset from TUDataset for classification with 4337 graphs. The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation.

| model | kgcnn | epochs | Accuracy | AUC(ROC) | 
| :---: | :---: | :---: | :---: | :---: | 
| AttentiveFP | 2.1.0 | 200 | 0.7466 &pm; 0.0216  | 0.8274 &pm; 0.0187  | 
| CMPNN | 2.1.0 | 600 | 0.8098 &pm; 0.0068  | 0.8331 &pm; 0.0070  | 
| DMPNN | 2.1.0 | 300 | 0.8271 &pm; 0.0069  | 0.8685 &pm; 0.0133  | 
| GAT | 2.1.0 | 500 | 0.7902 &pm; 0.0125  | 0.8469 &pm; 0.0117  | 
| GATv2 | 2.1.0 | 500 | 0.8084 &pm; 0.0130  | 0.8320 &pm; 0.0116  | 
| GIN | 2.1.0 | 300 | 0.8262 &pm; 0.0110  | 0.8818 &pm; 0.0045  | 
| GraphSAGE | 2.1.0 | 500 | 0.8063 &pm; 0.0097  | 0.8449 &pm; 0.0147  | 
| INorp | 2.1.0 | 500 | 0.8040 &pm; 0.0113  | 0.8290 &pm; 0.0117  | 

## MUTAGDataset

MUTAG dataset from TUDataset for classification with 188 graphs. We use random 5-fold cross-validation.

| model | kgcnn | epochs | Accuracy | AUC(ROC) | 
| :---: | :---: | :---: | :---: | :---: | 
| AttentiveFP | 2.1.0 | 200 | 0.8455 &pm; 0.0600  | 0.8893 &pm; 0.0812  | 
| CMPNN | 2.1.0 | 600 | 0.8138 &pm; 0.0612  | 0.8133 &pm; 0.0680  | 
| DMPNN | 2.1.0 | 300 | 0.8506 &pm; 0.0447  | 0.9038 &pm; 0.0435  | 
| GAT | 2.1.0 | 500 | 0.8141 &pm; 0.0405  | 0.8698 &pm; 0.0499  | 
| GATv2 | 2.1.0 | 500 | 0.7660 &pm; 0.0303  | 0.7885 &pm; 0.0433  | 
| GIN | 2.1.0 | 300 | 0.8243 &pm; 0.0372  | 0.8570 &pm; 0.0422  | 
| GraphSAGE | 2.1.0 | 500 | 0.8512 &pm; 0.0263  | 0.8707 &pm; 0.0449  | 
| INorp | 2.1.0 | 500 | 0.8450 &pm; 0.0682  | 0.8519 &pm; 0.1071  | 

