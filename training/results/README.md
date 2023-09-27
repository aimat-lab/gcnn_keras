# Summary of Benchmark Training

Note that these are the results for models within `kgcnn` implementation, and that training is not always done with optimal hyperparameter or splits, when comparing with literature.
This table is generated automatically from keras history logs.
Model weights and training statistics plots are not uploaded on 
[github](https://github.com/aimat-lab/gcnn_keras/tree/master/training/results) 
due to their file size.

*Max.* or *Min.* denotes the best test error observed for any epoch during training.
To show overall best test error run ``python3 summary.py --min_max True``.
If not noted otherwise, we use a (fixed) random k-fold split for validation errors.

#### CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse attributes and 7 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   |
|:----------|:--------|---------:|:-----------------------|
| GAT       | 4.0.0   |      250 | 0.8464 &pm; 0.0105     |
| GATv2     | 4.0.0   |      250 | 0.8331 &pm; 0.0104     |
| GCN       | 4.0.0   |      300 | 0.8072 &pm; 0.0109     |
| GIN       | 4.0.0   |      500 | 0.8279 &pm; 0.0170     |
| GraphSAGE | 4.0.0   |      500 | **0.8497 &pm; 0.0100** |

#### CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes. 

| model   | kgcnn   |   epochs | Categorical accuracy   |
|:--------|:--------|---------:|:-----------------------|
| GCN     | 4.0.0   |      300 | **0.6232 &pm; 0.0054** |

#### ESOLDataset

ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation. 

| model   | kgcnn   |   epochs | MAE [log mol/L]    | RMSE [log mol/L]   |
|:--------|:--------|---------:|:-------------------|:-------------------|
| DMPNN   | 4.0.0   |      300 | 0.4556 &pm; 0.0281 | 0.6471 &pm; 0.0299 |
| GAT     | 4.0.0   |      500 | **nan &pm; nan**   | **nan &pm; nan**   |
| GCN     | 4.0.0   |      800 | 0.4613 &pm; 0.0205 | 0.6534 &pm; 0.0513 |
| Schnet  | 4.0.0   |      800 | nan &pm; nan       | nan &pm; nan       |

#### MatProjectJdft2dDataset

Materials Project dataset from Matbench with 636 crystal structures and their corresponding Exfoliation energy (meV/atom). We use a random 5-fold cross-validation. 

| model                     | kgcnn   |   epochs | MAE [meV/atom]           | RMSE [meV/atom]           |
|:--------------------------|:--------|---------:|:-------------------------|:--------------------------|
| Schnet.make_crystal_model | 4.0.0   |      800 | **47.0970 &pm; 12.1636** | **121.0402 &pm; 38.7995** |

#### MD17Dataset

Energies and forces for molecular dynamics trajectories of eight organic molecules. All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. Errors are MAE for forces. Results are for the CCSD and CCSD(T) data in MD17. 

| model                   | kgcnn   |   epochs | Aspirin             | Toluene             | Malonaldehyde       | Benzene             | Ethanol             |
|:------------------------|:--------|---------:|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|
| Schnet.EnergyForceModel | 4.0.0   |     1000 | **1.2173 &pm; nan** | **0.7395 &pm; nan** | **0.8444 &pm; nan** | **0.3429 &pm; nan** | **0.5471 &pm; nan** |

