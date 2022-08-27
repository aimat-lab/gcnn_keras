# Summary of Benchmark Training

Note that these are the results for models within `kgcnn`, and that training is not always done with optimal hyperparameter, when comparing with literature.

## CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse node attributes and 7 node classes.

| model | kgcnn | epochs | Accuracy | 
| :---: | :---: | :---: | :---: | 
| GAT | 2.1.0 | 250 | 0.8645 &pm; 0.0129  | 
| GCN | 2.1.0 | 300 | 0.7969 &pm; 0.0144  | 

## CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes.

| model | kgcnn | epochs | Accuracy | 
| :---: | :---: | :---: | :---: | 
| GCN | 2.1.0 | 300 | 0.6150 &pm; 0.0121  | 

## ESOLDataset

ESOL (MoleculeNet) consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L).

| model | kgcnn | epochs | MAE [log mol/L] | RMSE [log mol/L] | 
| :---: | :---: | :---: | :---: | :---: | 
| AttentiveFP | 2.1.0 | 200 | 0.4272 &pm; 0.0297  | 0.6002 &pm; 0.0268  | 
| CMPNN | 2.1.0 | 300 | 0.4318 &pm; 0.0173  | 0.5919 &pm; 0.0262  | 
| DMPNN | 2.1.0 | 300 | 0.4463 &pm; 0.0357  | 0.6442 &pm; 0.0429  | 
| GAT | 2.1.0 | 500 | 0.4699 &pm; 0.0435  | 0.6711 &pm; 0.0745  | 
| GATv2 | 2.1.0 | 500 | 0.4628 &pm; 0.0432  | 0.6615 &pm; 0.0565  | 
| GraphSAGE | 2.1.0 | 500 | 0.4896 &pm; 0.0169  | 0.7003 &pm; 0.0494  | 

