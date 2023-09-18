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
| GIN       | 4.0.0   |      500 | 0.8213 &pm; 0.0233     |
| GraphSAGE | 4.0.0   |      500 | **0.8497 &pm; 0.0100** |

#### ESOLDataset

ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation. 

| model   | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:--------|:--------|---------:|:-----------------------|:-----------------------|
| GAT     | 4.0.0   |      500 | 0.4826 &pm; 0.0255     | 0.6903 &pm; 0.0705     |
| GCN     | 4.0.0   |      800 | **0.4623 &pm; 0.0224** | **0.6567 &pm; 0.0456** |
| Schnet  | 4.0.0   |      800 | 0.4678 &pm; 0.0227     | 0.6662 &pm; 0.0629     |

