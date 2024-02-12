# Summary of Benchmark Training

Note that these are the results for models within `kgcnn` implementation, and that training is not always done with optimal hyperparameter or splits, when comparing with literature.
This table is generated automatically from keras history logs.
Model weights and training statistics plots are not uploaded on 
[github](https://github.com/aimat-lab/gcnn_keras/tree/master/training/results) 
due to their file size.

*Max.* or *Min.* denotes the best test error observed for any epoch during training.
To show overall best test error run ``python3 summary.py --min_max True``.
If not noted otherwise, we use a (fixed) random k-fold split for validation errors.

#### ClinToxDataset

ClinTox (MoleculeNet) consists of 1478 compounds as smiles and data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons. We use random 5-fold cross-validation. The first label 'approved' is chosen as target.

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |       50 | 0.9480 &pm; 0.0138     | 0.8297 &pm; 0.0568     | **0.9594 &pm; 0.0071** | 0.8928 &pm; 0.0301     |
| GAT       | 4.0.0   |       50 | **0.9480 &pm; 0.0070** | 0.8512 &pm; 0.0468     | 0.9561 &pm; 0.0077     | 0.8740 &pm; 0.0436     |
| GATv2     | 4.0.0   |       50 | 0.9372 &pm; 0.0155     | **0.8587 &pm; 0.0754** | 0.9581 &pm; 0.0102     | 0.8915 &pm; 0.0539     |
| GCN       | 4.0.0   |       50 | 0.9432 &pm; 0.0155     | 0.8555 &pm; 0.0593     | 0.9574 &pm; 0.0082     | 0.8876 &pm; 0.0378     |
| GIN       | 4.0.0   |       50 | 0.9412 &pm; 0.0034     | 0.8066 &pm; 0.0636     | 0.9567 &pm; 0.0102     | 0.8634 &pm; 0.0482     |
| GraphSAGE | 4.0.0   |      100 | 0.9412 &pm; 0.0073     | 0.8013 &pm; 0.0422     | 0.9547 &pm; 0.0076     | **0.8933 &pm; 0.0411** |
| Schnet    | 4.0.0   |       50 | 0.9277 &pm; 0.0102     | 0.6562 &pm; 0.0760     | 0.9392 &pm; 0.0125     | 0.7721 &pm; 0.0510     |

#### CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   | *Max. Categorical accuracy*   |
|:----------|:--------|---------:|:-----------------------|:------------------------------|
| DMPNN     | 4.0.0   |      300 | 0.2476 &pm; 0.1706     | 0.2554 &pm; 0.1643            |
| GAT       | 4.0.0   |      250 | 0.6157 &pm; 0.0071     | 0.6331 &pm; 0.0089            |
| GATv2     | 4.0.0   |     1000 | 0.6211 &pm; 0.0048     | 0.6383 &pm; 0.0079            |
| GCN       | 4.0.0   |      300 | 0.6232 &pm; 0.0054     | 0.6307 &pm; 0.0061            |
| GIN       | 4.0.0   |      800 | **0.6263 &pm; 0.0080** | 0.6323 &pm; 0.0087            |
| GraphSAGE | 4.0.0   |      600 | 0.6151 &pm; 0.0053     | **0.6431 &pm; 0.0027**        |

#### CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse attributes and 7 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   | *Max. Categorical accuracy*   |
|:----------|:--------|---------:|:-----------------------|:------------------------------|
| DMPNN     | 4.0.0   |      300 | 0.8357 &pm; 0.0156     | 0.8545 &pm; 0.0181            |
| GAT       | 4.0.0   |      250 | 0.8397 &pm; 0.0122     | 0.8512 &pm; 0.0147            |
| GATv2     | 4.0.0   |      250 | 0.8331 &pm; 0.0104     | 0.8427 &pm; 0.0120            |
| GCN       | 4.0.0   |      300 | 0.8072 &pm; 0.0109     | 0.8497 &pm; 0.0149            |
| GIN       | 4.0.0   |      500 | 0.8279 &pm; 0.0170     | 0.8335 &pm; 0.0176            |
| GraphSAGE | 4.0.0   |      500 | **0.8497 &pm; 0.0100** | **0.8741 &pm; 0.0115**        |

#### ESOLDataset

ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       | *Min. MAE*             | *Min. RMSE*            |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 4.0.0   |      200 | 0.4351 &pm; 0.0110     | 0.6080 &pm; 0.0207     | **0.4023 &pm; 0.0185** | **0.5633 &pm; 0.0328** |
| CMPNN       | 4.0.0   |      600 | 0.5276 &pm; 0.0154     | 0.7505 &pm; 0.0189     | 0.4681 &pm; 0.0107     | 0.6351 &pm; 0.0182     |
| DGIN        | 4.0.0   |      300 | 0.4434 &pm; 0.0252     | 0.6225 &pm; 0.0420     | 0.4247 &pm; 0.0180     | 0.5980 &pm; 0.0277     |
| DMPNN       | 4.0.0   |      300 | 0.4401 &pm; 0.0165     | 0.6203 &pm; 0.0292     | 0.4261 &pm; 0.0118     | 0.5968 &pm; 0.0211     |
| EGNN        | 4.0.0   |      800 | 0.4507 &pm; 0.0152     | 0.6563 &pm; 0.0370     | 0.4209 &pm; 0.0129     | 0.5977 &pm; 0.0444     |
| GAT         | 4.0.0   |      500 | 0.4818 &pm; 0.0240     | 0.6919 &pm; 0.0694     | 0.4550 &pm; 0.0230     | 0.6491 &pm; 0.0591     |
| GATv2       | 4.0.0   |      500 | 0.4598 &pm; 0.0234     | 0.6650 &pm; 0.0409     | 0.4372 &pm; 0.0217     | 0.6217 &pm; 0.0450     |
| GCN         | 4.0.0   |      800 | 0.4613 &pm; 0.0205     | 0.6534 &pm; 0.0513     | 0.4405 &pm; 0.0277     | 0.6197 &pm; 0.0602     |
| GIN         | 4.0.0   |      300 | 0.5369 &pm; 0.0334     | 0.7954 &pm; 0.0861     | 0.4967 &pm; 0.0159     | 0.7332 &pm; 0.0647     |
| GNNFilm     | 4.0.0   |      800 | 0.4854 &pm; 0.0368     | 0.6724 &pm; 0.0436     | 0.4669 &pm; 0.0317     | 0.6488 &pm; 0.0370     |
| GraphSAGE   | 4.0.0   |      500 | 0.4874 &pm; 0.0228     | 0.6982 &pm; 0.0608     | 0.4774 &pm; 0.0239     | 0.6789 &pm; 0.0521     |
| HamNet      | 4.0.0   |      400 | 0.5479 &pm; 0.0143     | 0.7417 &pm; 0.0298     | 0.5109 &pm; 0.0112     | 0.7008 &pm; 0.0241     |
| HDNNP2nd    | 4.0.0   |      500 | 0.7857 &pm; 0.0986     | 1.0467 &pm; 0.1367     | 0.7620 &pm; 0.1024     | 1.0097 &pm; 0.1326     |
| INorp       | 4.0.0   |      500 | 0.5055 &pm; 0.0436     | 0.7297 &pm; 0.0786     | 0.4791 &pm; 0.0348     | 0.6687 &pm; 0.0520     |
| MAT         | 4.0.0   |      400 | 0.5064 &pm; 0.0299     | 0.7194 &pm; 0.0630     | 0.5035 &pm; 0.0288     | 0.7125 &pm; 0.0570     |
| MEGAN       | 4.0.0   |      400 | **0.4281 &pm; 0.0201** | **0.6062 &pm; 0.0252** | 0.4161 &pm; 0.0139     | 0.5798 &pm; 0.0201     |
| Megnet      | 4.0.0   |      800 | 0.5679 &pm; 0.0310     | 0.8196 &pm; 0.0480     | 0.5059 &pm; 0.0258     | 0.7003 &pm; 0.0454     |
| MoGAT       | 4.0.0   |      200 | 0.4797 &pm; 0.0114     | 0.6533 &pm; 0.0114     | 0.4613 &pm; 0.0135     | 0.6247 &pm; 0.0161     |
| MXMNet      | 4.0.0   |      900 | 0.6486 &pm; 0.0633     | 1.0123 &pm; 0.2059     | 0.6008 &pm; 0.0575     | 0.8923 &pm; 0.1685     |
| NMPN        | 4.0.0   |      800 | 0.5046 &pm; 0.0266     | 0.7193 &pm; 0.0607     | 0.4823 &pm; 0.0226     | 0.6729 &pm; 0.0521     |
| PAiNN       | 4.0.0   |      250 | 0.4857 &pm; 0.0598     | 0.6650 &pm; 0.0674     | 0.4206 &pm; 0.0157     | 0.5925 &pm; 0.0476     |
| RGCN        | 4.0.0   |      800 | 0.4703 &pm; 0.0251     | 0.6529 &pm; 0.0318     | 0.4387 &pm; 0.0178     | 0.6048 &pm; 0.0240     |
| rGIN        | 4.0.0   |      300 | 0.5196 &pm; 0.0351     | 0.7142 &pm; 0.0263     | 0.4956 &pm; 0.0292     | 0.6887 &pm; 0.0231     |
| Schnet      | 4.0.0   |      800 | 0.4777 &pm; 0.0294     | 0.6977 &pm; 0.0538     | 0.4503 &pm; 0.0243     | 0.6416 &pm; 0.0434     |

#### FreeSolvDataset

FreeSolv (MoleculeNet) consists of 642 compounds as smiles and their corresponding hydration free energy for small neutral molecules in water. We use a random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       | *Min. MAE*             | *Min. RMSE*            |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CMPNN     | 4.0.0   |      600 | 0.5202 &pm; 0.0504     | 0.9339 &pm; 0.1286     | 0.5016 &pm; 0.0551     | 0.8886 &pm; 0.1249     |
| DGIN      | 4.0.0   |      300 | 0.5489 &pm; 0.0374     | 0.9448 &pm; 0.0787     | 0.5132 &pm; 0.0452     | 0.8704 &pm; 0.1177     |
| DimeNetPP | 4.0.0   |      872 | 0.6167 &pm; 0.0719     | 1.0302 &pm; 0.1717     | 0.5907 &pm; 0.0663     | 0.9580 &pm; 0.1503     |
| DMPNN     | 4.0.0   |      300 | 0.5487 &pm; 0.0754     | **0.9206 &pm; 0.1889** | **0.4947 &pm; 0.0665** | **0.8362 &pm; 0.1812** |
| EGNN      | 4.0.0   |      800 | 0.5386 &pm; 0.0548     | 1.0363 &pm; 0.1237     | 0.5268 &pm; 0.0607     | 0.9849 &pm; 0.1590     |
| GAT       | 4.0.0   |      500 | 0.6051 &pm; 0.0861     | 1.0326 &pm; 0.1819     | 0.5790 &pm; 0.0880     | 0.9717 &pm; 0.2008     |
| GATv2     | 4.0.0   |      500 | 0.6151 &pm; 0.0247     | 1.0535 &pm; 0.0817     | 0.5971 &pm; 0.0177     | 1.0037 &pm; 0.0753     |
| GCN       | 4.0.0   |      800 | 0.6400 &pm; 0.0834     | 1.0876 &pm; 0.1393     | 0.5780 &pm; 0.0836     | 0.9438 &pm; 0.1597     |
| GIN       | 4.0.0   |      300 | 0.8100 &pm; 0.1016     | 1.2695 &pm; 0.1192     | 0.6720 &pm; 0.0516     | 1.0699 &pm; 0.0662     |
| GNNFilm   | 4.0.0   |      800 | 0.6562 &pm; 0.0552     | 1.1597 &pm; 0.1245     | 0.6358 &pm; 0.0606     | 1.1168 &pm; 0.1371     |
| GraphSAGE | 4.0.0   |      500 | 0.5894 &pm; 0.0675     | 1.0009 &pm; 0.1491     | 0.5700 &pm; 0.0615     | 0.9508 &pm; 0.1333     |
| HamNet    | 4.0.0   |      400 | 0.6619 &pm; 0.0428     | 1.1410 &pm; 0.1120     | 0.6005 &pm; 0.0466     | 1.0120 &pm; 0.0800     |
| HDNNP2nd  | 4.0.0   |      500 | 1.0201 &pm; 0.1559     | 1.6351 &pm; 0.3419     | 0.9933 &pm; 0.1523     | 1.5395 &pm; 0.2969     |
| INorp     | 4.0.0   |      500 | 0.6612 &pm; 0.0188     | 1.1155 &pm; 0.1061     | 0.6391 &pm; 0.0154     | 1.0556 &pm; 0.1064     |
| MAT       | 4.0.0   |      400 | 0.8115 &pm; 0.0649     | 1.3099 &pm; 0.1235     | 0.7915 &pm; 0.0687     | 1.2256 &pm; 0.1712     |
| MEGAN     | 4.0.0   |      400 | 0.6303 &pm; 0.0550     | 1.0429 &pm; 0.1031     | 0.6141 &pm; 0.0540     | 1.0192 &pm; 0.1074     |
| Megnet    | 4.0.0   |      800 | 0.8878 &pm; 0.0528     | 1.4134 &pm; 0.1200     | 0.8090 &pm; 0.0405     | 1.2735 &pm; 0.1157     |
| MoGAT     | 4.0.0   |      200 | 0.7097 &pm; 0.0374     | 1.0911 &pm; 0.1334     | 0.6596 &pm; 0.0450     | 1.0424 &pm; 0.1313     |
| MXMNet    | 4.0.0   |      900 | 1.1386 &pm; 0.1979     | 3.0487 &pm; 2.1757     | 1.0970 &pm; 0.1909     | 2.8598 &pm; 2.0855     |
| RGCN      | 4.0.0   |      800 | **0.5128 &pm; 0.0810** | 0.9228 &pm; 0.1887     | 0.4956 &pm; 0.0864     | 0.8678 &pm; 0.2111     |
| rGIN      | 4.0.0   |      300 | 0.8503 &pm; 0.0613     | 1.3285 &pm; 0.0976     | 0.8042 &pm; 0.0777     | 1.2469 &pm; 0.1013     |
| Schnet    | 4.0.0   |      800 | 0.6070 &pm; 0.0285     | 1.0603 &pm; 0.0549     | 0.5688 &pm; 0.0314     | 0.9526 &pm; 0.0840     |

#### ISO17Dataset

The database consist of 129 molecules each containing 5,000 conformational geometries, energies and forces with a resolution of 1 femtosecond in the molecular dynamics trajectories. The molecules were randomly drawn from the largest set of isomers in the QM9 dataset. 

| model                   | kgcnn   |   epochs | Energy (test_within)   | Force (test_within)   | *Min. Energy* (test_within)   | *Min. Force* (test_within)   |
|:------------------------|:--------|---------:|:-----------------------|:----------------------|:------------------------------|:-----------------------------|
| Schnet.EnergyForceModel | 4.0.0   |     1000 | **0.0061 &pm; nan**    | **0.0134 &pm; nan**   | **0.0057 &pm; nan**           | **0.0134 &pm; nan**          |

#### LipopDataset

Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles. Graph labels for regression are octanol/water distribution coefficient (logD at pH 7.4). We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       | *Min. MAE*             | *Min. RMSE*            |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |      300 | **0.3814 &pm; 0.0064** | **0.5462 &pm; 0.0095** | **0.3774 &pm; 0.0072** | **0.5421 &pm; 0.0093** |
| GAT       | 4.0.0   |      500 | 0.5168 &pm; 0.0088     | 0.7220 &pm; 0.0098     | 0.4906 &pm; 0.0092     | 0.6819 &pm; 0.0079     |
| GATv2     | 4.0.0   |      500 | 0.4342 &pm; 0.0104     | 0.6056 &pm; 0.0114     | 0.4163 &pm; 0.0089     | 0.5785 &pm; 0.0163     |
| GCN       | 4.0.0   |      800 | 0.4960 &pm; 0.0107     | 0.6833 &pm; 0.0155     | 0.4729 &pm; 0.0126     | 0.6496 &pm; 0.0116     |
| GIN       | 4.0.0   |      300 | 0.4745 &pm; 0.0101     | 0.6658 &pm; 0.0159     | 0.4703 &pm; 0.0089     | 0.6555 &pm; 0.0163     |
| GraphSAGE | 4.0.0   |      500 | 0.4333 &pm; 0.0217     | 0.6218 &pm; 0.0318     | 0.4296 &pm; 0.0175     | 0.6108 &pm; 0.0258     |
| Schnet    | 4.0.0   |      800 | 0.5657 &pm; 0.0202     | 0.7485 &pm; 0.0245     | 0.5280 &pm; 0.0136     | 0.7024 &pm; 0.0210     |

#### MD17Dataset

Energies and forces for molecular dynamics trajectories of eight organic molecules. All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. Errors are MAE for forces. Results are for the CCSD and CCSD(T) data in MD17. 

| model                   | kgcnn   |   epochs | Aspirin             | Toluene             | Malonaldehyde       | Benzene             | Ethanol             |
|:------------------------|:--------|---------:|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|
| PAiNN.EnergyForceModel  | 4.0.0   |     1000 | **0.8551 &pm; nan** | **0.2815 &pm; nan** | **0.7749 &pm; nan** | **0.0427 &pm; nan** | 0.5805 &pm; nan     |
| Schnet.EnergyForceModel | 4.0.0   |     1000 | 1.2173 &pm; nan     | 0.7395 &pm; nan     | 0.8444 &pm; nan     | 0.3353 &pm; nan     | **0.4832 &pm; nan** |

#### MD17RevisedDataset

Energies and forces for molecular dynamics trajectories. All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. Errors are MAE for forces. 

| model                   | kgcnn   |   epochs | Aspirin                | Toluene                | Malonaldehyde          | Benzene                | Ethanol                |
|:------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| Schnet.EnergyForceModel | 4.0.0   |     1000 | **1.0389 &pm; 0.0071** | **0.5482 &pm; 0.0105** | **0.6727 &pm; 0.0132** | **0.2525 &pm; 0.0091** | **0.4471 &pm; 0.0199** |

#### MatProjectDielectricDataset

Materials Project dataset from Matbench with 4764 crystal structures and their corresponding Refractive index (unitless). We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [no unit]          | RMSE [no unit]         | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 0.3306 &pm; 0.0602     | 1.9736 &pm; 0.7324     | 0.3012 &pm; 0.0561     | 1.7712 &pm; 0.6468     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 0.3415 &pm; 0.0542     | 1.9637 &pm; 0.6323     | 0.3031 &pm; 0.0526     | 1.7761 &pm; 0.6535     |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 0.3362 &pm; 0.0550     | 2.0156 &pm; 0.5872     | 0.3007 &pm; 0.0563     | 1.7416 &pm; 0.6413     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 0.3289 &pm; 0.0489     | 1.8770 &pm; 0.6522     | 0.3037 &pm; 0.0485     | 1.7718 &pm; 0.6470     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 0.3539 &pm; 0.0433     | 1.8661 &pm; 0.5984     | 0.3063 &pm; 0.0481     | 1.7823 &pm; 0.6299     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | **0.3180 &pm; 0.0359** | **1.8509 &pm; 0.5854** | **0.2914 &pm; 0.0475** | **1.7244 &pm; 0.6188** |

#### MatProjectEFormDataset

Materials Project dataset from Matbench with 132752 crystal structures and their corresponding formation energy in [eV/atom]. We use a random 5-fold cross-validation. 

| model                     | kgcnn   |   epochs | MAE [eV/atom]          | RMSE [eV/atom]         | *Min. MAE*             | *Min. RMSE*            |
|:--------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model  | 4.0.0   |     1000 | 0.0298 &pm; 0.0002     | 0.0747 &pm; 0.0029     | 0.0298 &pm; 0.0002     | 0.0738 &pm; 0.0029     |
| Schnet.make_crystal_model | 4.0.0   |      800 | **0.0211 &pm; 0.0003** | **0.0510 &pm; 0.0024** | **0.0211 &pm; 0.0003** | **0.0505 &pm; 0.0023** |

#### MatProjectGapDataset

Materials Project dataset from Matbench with 106113 crystal structures and their band gap as calculated by PBE DFT from the Materials Project, in eV. We use a random 5-fold cross-validation. 

| model                     | kgcnn   |   epochs | MAE [eV]               | RMSE [eV]              | *Min. MAE*             | *Min. RMSE*            |
|:--------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model  | 4.0.0   |     1000 | **0.2039 &pm; 0.0050** | **0.4882 &pm; 0.0213** | **0.2039 &pm; 0.0050** | **0.4783 &pm; 0.0203** |
| Schnet.make_crystal_model | 4.0.0   |      800 | 1.2226 &pm; 1.0573     | 58.3713 &pm; 114.2957  | 0.2983 &pm; 0.0257     | 0.6192 &pm; 0.0409     |

#### MatProjectIsMetalDataset

Materials Project dataset from Matbench with 106113 crystal structures and their corresponding Metallicity determined with pymatgen. 1 if the compound is a metal, 0 if the compound is not a metal. We use a random 5-fold cross-validation. 

| model                     | kgcnn   |   epochs | Accuracy               | AUC                    | *Max. Accuracy*        | *Max. AUC*       |
|:--------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------|
| CGCNN.make_crystal_model  | 4.0.0   |      100 | 0.8910 &pm; 0.0027     | 0.9406 &pm; 0.0024     | 0.8954 &pm; 0.0028     | **nan &pm; nan** |
| Megnet.make_crystal_model | 4.0.0   |      100 | **0.8966 &pm; 0.0033** | **0.9506 &pm; 0.0026** | 0.8995 &pm; 0.0027     | nan &pm; nan     |
| Schnet.make_crystal_model | 4.0.0   |       80 | 0.8953 &pm; 0.0058     | 0.9506 &pm; 0.0053     | **0.9005 &pm; 0.0027** | nan &pm; nan     |

#### MatProjectJdft2dDataset

Materials Project dataset from Matbench with 636 crystal structures and their corresponding Exfoliation energy (meV/atom). We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [meV/atom]           | RMSE [meV/atom]           | *Min. MAE*               | *Min. RMSE*               |
|:-----------------------------|:--------|---------:|:-------------------------|:--------------------------|:-------------------------|:--------------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 57.6974 &pm; 18.0803     | 140.6167 &pm; 44.8418     | 46.6901 &pm; 13.5301     | 121.0725 &pm; 44.0067     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 50.2880 &pm; 11.4199     | 126.0600 &pm; 38.3769     | 46.1936 &pm; 11.8615     | 118.6555 &pm; 38.6340     |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 51.1735 &pm; 9.1746      | 123.4178 &pm; 32.9582     | 45.2357 &pm; 10.1934     | 113.8528 &pm; 37.2491     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 59.3986 &pm; 10.9272     | 139.5943 &pm; 32.1129     | 48.0720 &pm; 12.1130     | 120.6016 &pm; 39.6981     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 49.3889 &pm; 11.5376     | 121.7087 &pm; 30.0472     | 46.6649 &pm; 11.5589     | 117.9086 &pm; 32.8603     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | **45.2412 &pm; 11.6395** | **115.6890 &pm; 39.0929** | **41.4056 &pm; 10.7214** | **112.5666 &pm; 38.0183** |

#### MatProjectLogGVRHDataset

Materials Project dataset from Matbench with 10987 crystal structures and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average shear moduli in GPa. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [log(GPa)]         | RMSE [log(GPa)]        | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 0.0874 &pm; 0.0022     | 0.1354 &pm; 0.0056     | 0.0870 &pm; 0.0018     | 0.1316 &pm; 0.0041     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 0.0839 &pm; 0.0027     | **0.1290 &pm; 0.0065** | **0.0809 &pm; 0.0024** | **0.1232 &pm; 0.0049** |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 0.0885 &pm; 0.0017     | 0.1360 &pm; 0.0054     | 0.0883 &pm; 0.0016     | 0.1342 &pm; 0.0049     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 0.0874 &pm; 0.0027     | 0.1324 &pm; 0.0045     | 0.0867 &pm; 0.0025     | 0.1310 &pm; 0.0040     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 0.0870 &pm; 0.0033     | 0.1332 &pm; 0.0103     | 0.0845 &pm; 0.0017     | 0.1254 &pm; 0.0046     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | **0.0836 &pm; 0.0021** | 0.1296 &pm; 0.0044     | 0.0828 &pm; 0.0020     | 0.1277 &pm; 0.0043     |

#### MatProjectLogKVRHDataset

Materials Project dataset from Matbench with 10987 crystal structures and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average bulk moduli in GPa. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [log(GPa)]         | RMSE [log(GPa)]        | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 0.0672 &pm; 0.0012     | 0.1265 &pm; 0.0042     | 0.0646 &pm; 0.0007     | 0.1199 &pm; 0.0036     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | **0.0604 &pm; 0.0023** | **0.1141 &pm; 0.0055** | 0.0588 &pm; 0.0019     | 0.1095 &pm; 0.0057     |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 0.0686 &pm; 0.0016     | 0.1285 &pm; 0.0061     | 0.0675 &pm; 0.0013     | 0.1264 &pm; 0.0052     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 0.0688 &pm; 0.0009     | 0.1262 &pm; 0.0031     | 0.0647 &pm; 0.0015     | 0.1189 &pm; 0.0042     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 0.0649 &pm; 0.0007     | 0.1170 &pm; 0.0048     | **0.0565 &pm; 0.0009** | **0.1080 &pm; 0.0045** |
| Schnet.make_crystal_model    | 4.0.0   |      800 | 0.0635 &pm; 0.0016     | 0.1186 &pm; 0.0044     | 0.0629 &pm; 0.0013     | 0.1154 &pm; 0.0046     |

#### MatProjectPerovskitesDataset

Materials Project dataset from Matbench with 18928 crystal structures and their corresponding Heat of formation of the entire 5-atom perovskite cell in eV. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV]               | RMSE [eV]              | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 0.0425 &pm; 0.0011     | 0.0712 &pm; 0.0037     | 0.0422 &pm; 0.0015     | 0.0684 &pm; 0.0030     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 0.0447 &pm; 0.0016     | 0.0730 &pm; 0.0050     | 0.0415 &pm; 0.0015     | 0.0690 &pm; 0.0045     |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 0.0388 &pm; 0.0017     | 0.0675 &pm; 0.0041     | 0.0388 &pm; 0.0017     | 0.0675 &pm; 0.0041     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | **0.0381 &pm; 0.0009** | 0.0652 &pm; 0.0029     | **0.0380 &pm; 0.0009** | 0.0649 &pm; 0.0029     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 0.0474 &pm; 0.0003     | 0.0762 &pm; 0.0017     | 0.0472 &pm; 0.0004     | 0.0759 &pm; 0.0017     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | 0.0381 &pm; 0.0005     | **0.0645 &pm; 0.0024** | 0.0380 &pm; 0.0005     | **0.0644 &pm; 0.0022** |

#### MatProjectPhononsDataset

Materials Project dataset from Matbench with 1,265 crystal structures and their corresponding vibration properties in [1/cm]. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV/atom]           | RMSE [eV/atom]           | *Min. MAE*              | *Min. RMSE*              |
|:-----------------------------|:--------|---------:|:------------------------|:-------------------------|:------------------------|:-------------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 42.6447 &pm; 4.5721     | 92.1627 &pm; 21.4345     | 41.3049 &pm; 3.8502     | 86.9412 &pm; 16.6723     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 39.8893 &pm; 3.1280     | 77.5776 &pm; 16.0908     | 36.1806 &pm; 2.1331     | 67.9898 &pm; 7.9298      |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | **30.6620 &pm; 2.9013** | **60.8733 &pm; 17.1448** | **28.9268 &pm; 3.0908** | **54.5838 &pm; 13.5562** |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 45.9344 &pm; 5.7908     | 95.4136 &pm; 35.5401     | 43.0340 &pm; 4.1057     | 79.5178 &pm; 28.0048     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 47.5408 &pm; 4.2815     | 86.6761 &pm; 11.9220     | 45.9714 &pm; 3.3346     | 79.7746 &pm; 8.6082      |
| Schnet.make_crystal_model    | 4.0.0   |      800 | 43.0692 &pm; 3.6227     | 88.5151 &pm; 20.0244     | 41.8227 &pm; 3.4578     | 76.7519 &pm; 16.4611     |

#### MUTAGDataset

MUTAG dataset from TUDataset for classification with 188 graphs. We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |      300 | **0.8407 &pm; 0.0463** | 0.8567 &pm; 0.0511     | 0.9098 &pm; 0.0390     | **0.9564 &pm; 0.0243** |
| GAT       | 4.0.0   |      500 | 0.8141 &pm; 0.1077     | 0.8671 &pm; 0.0923     | 0.8407 &pm; 0.0926     | 0.9402 &pm; 0.0542     |
| GATv2     | 4.0.0   |      500 | 0.8193 &pm; 0.0945     | 0.8379 &pm; 0.1074     | 0.8248 &pm; 0.0976     | 0.9360 &pm; 0.0512     |
| GCN       | 4.0.0   |      800 | 0.7716 &pm; 0.0531     | 0.7956 &pm; 0.0909     | 0.8673 &pm; 0.0573     | 0.9324 &pm; 0.0544     |
| GIN       | 4.0.0   |      300 | 0.8091 &pm; 0.0781     | **0.8693 &pm; 0.0855** | **0.9100 &pm; 0.0587** | 0.9539 &pm; 0.0564     |
| GraphSAGE | 4.0.0   |      500 | 0.8357 &pm; 0.0798     | 0.8533 &pm; 0.0824     | 0.8886 &pm; 0.0710     | 0.8957 &pm; 0.0814     |

#### MutagenicityDataset

Mutagenicity dataset from TUDataset for classification with 4337 graphs. The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |      300 | **0.8266 &pm; 0.0059** | **0.8708 &pm; 0.0076** | **0.8423 &pm; 0.0073** | **0.8968 &pm; 0.0109** |
| GAT       | 4.0.0   |      500 | 0.7989 &pm; 0.0114     | 0.8290 &pm; 0.0112     | 0.8119 &pm; 0.0049     | 0.8700 &pm; 0.0077     |
| GATv2     | 4.0.0   |      200 | 0.7674 &pm; 0.0048     | 0.8423 &pm; 0.0064     | 0.7743 &pm; 0.0079     | 0.8426 &pm; 0.0062     |
| GCN       | 4.0.0   |      800 | 0.7955 &pm; 0.0154     | 0.8191 &pm; 0.0137     | 0.8130 &pm; 0.0090     | 0.8670 &pm; 0.0068     |
| GIN       | 4.0.0   |      300 | 0.8118 &pm; 0.0091     | 0.8492 &pm; 0.0077     | 0.8248 &pm; 0.0089     | 0.8798 &pm; 0.0026     |
| GraphSAGE | 4.0.0   |      500 | 0.8195 &pm; 0.0126     | 0.8515 &pm; 0.0083     | 0.8294 &pm; 0.0123     | 0.8851 &pm; 0.0061     |

#### PROTEINSDataset

TUDataset of proteins that are classified as enzymes or non-enzymes. Nodes represent the amino acids of the protein. We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |      300 | 0.7287 &pm; 0.0253     | **0.7970 &pm; 0.0343** | 0.7790 &pm; 0.0190     | 0.8298 &pm; 0.0329     |
| GAT       | 4.0.0   |      500 | **0.7314 &pm; 0.0357** | 0.7899 &pm; 0.0468     | 0.7763 &pm; 0.0380     | 0.8269 &pm; 0.0367     |
| GATv2     | 4.0.0   |      500 | 0.6720 &pm; 0.0595     | 0.6850 &pm; 0.0938     | **0.7898 &pm; 0.0272** | 0.8273 &pm; 0.0304     |
| GCN       | 4.0.0   |      800 | 0.7017 &pm; 0.0303     | 0.7211 &pm; 0.0254     | 0.7790 &pm; 0.0301     | **0.8342 &pm; 0.0358** |
| GIN       | 4.0.0   |      150 | 0.7224 &pm; 0.0343     | 0.7905 &pm; 0.0528     | 0.7700 &pm; 0.0299     | 0.8096 &pm; 0.0409     |
| GraphSAGE | 4.0.0   |      500 | 0.7009 &pm; 0.0398     | 0.7263 &pm; 0.0453     | 0.7691 &pm; 0.0369     | 0.7991 &pm; 0.0353     |

#### QM7Dataset

QM7 dataset is a subset of GDB-13. Molecules of up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. We use dataset-specific 5-fold cross-validation. The atomization energies are given in kcal/mol and are ranging from -800 to -2000 kcal/mol). 

| model     | kgcnn   |   epochs | MAE [kcal/mol]         | RMSE [kcal/mol]        | *Min. MAE*             | *Min. RMSE*            |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DimeNetPP | 4.0.0   |      872 | 3.4639 &pm; 0.2003     | 7.5327 &pm; 1.8190     | 3.4575 &pm; 0.1917     | 7.4462 &pm; 1.7268     |
| EGNN      | 4.0.0   |      800 | 1.7300 &pm; 0.1336     | 5.1268 &pm; 2.5134     | 1.7022 &pm; 0.1210     | 5.0965 &pm; 2.4826     |
| Megnet    | 4.0.0   |      800 | 1.5180 &pm; 0.0802     | 3.0321 &pm; 0.1936     | 1.5148 &pm; 0.0805     | 2.9391 &pm; 0.1885     |
| MXMNet    | 4.0.0   |      900 | **1.2431 &pm; 0.0820** | **2.6694 &pm; 0.2584** | **1.1588 &pm; 0.0840** | **2.6014 &pm; 0.2272** |
| NMPN      | 4.0.0   |      500 | 7.2907 &pm; 0.9061     | 38.1446 &pm; 12.1445   | 7.2489 &pm; 0.8699     | 35.4767 &pm; 10.2318   |
| PAiNN     | 4.0.0   |      872 | 1.5765 &pm; 0.0742     | 5.2705 &pm; 2.2848     | 1.5428 &pm; 0.0675     | 5.1099 &pm; 2.0842     |
| Schnet    | 4.0.0   |      800 | 3.4313 &pm; 0.4757     | 10.8978 &pm; 7.3863    | 3.3606 &pm; 0.4927     | 9.8169 &pm; 6.3053     |

#### QM9Dataset

QM9 dataset of 134k stable small organic molecules made up of C,H,O,N,F. Labels include geometric, energetic, electronic, and thermodynamic properties. We use a random 5-fold cross-validation, but not all splits are evaluated for cheaper evaluation. Test errors are MAE and for energies are given in [eV]. 

| model   | kgcnn   |   epochs | HOMO [eV]          | LUMO [eV]              | U0 [eV]            | H [eV]             | G [eV]                 |
|:--------|:--------|---------:|:-------------------|:-----------------------|:-------------------|:-------------------|:-----------------------|
| Megnet  | 4.0.0   |      800 | **nan &pm; nan**   | 0.0407 &pm; 0.0009     | **nan &pm; nan**   | **nan &pm; nan**   | 0.0169 &pm; 0.0006     |
| PAiNN   | 4.0.0   |      872 | 0.0483 &pm; 0.0275 | **0.0268 &pm; 0.0002** | 0.0099 &pm; 0.0003 | 0.0101 &pm; 0.0003 | **0.0110 &pm; 0.0002** |
| Schnet  | 4.0.0   |      800 | 0.0402 &pm; 0.0007 | 0.0340 &pm; 0.0001     | 0.0142 &pm; 0.0002 | 0.0146 &pm; 0.0002 | 0.0143 &pm; 0.0002     |

#### SIDERDataset

SIDER (MoleculeNet) consists of 1427 compounds as smiles and data for adverse drug reactions (ADR), grouped into 27 system organ classes. We use random 5-fold cross-validation.

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |       50 | 0.7519 &pm; 0.0055     | **0.6280 &pm; 0.0173** | **0.7629 &pm; 0.0041** | **0.6336 &pm; 0.0167** |
| GAT       | 4.0.0   |       50 | **0.7595 &pm; 0.0034** | 0.6224 &pm; 0.0106     | 0.7616 &pm; 0.0015     | 0.6231 &pm; 0.0101     |
| GATv2     | 4.0.0   |       50 | 0.7548 &pm; 0.0052     | 0.6152 &pm; 0.0154     | 0.7602 &pm; 0.0036     | 0.6201 &pm; 0.0169     |
| GIN       | 4.0.0   |       50 | 0.7472 &pm; 0.0055     | 0.5995 &pm; 0.0058     | 0.7565 &pm; 0.0032     | 0.6106 &pm; 0.0085     |
| GraphSAGE | 4.0.0   |       30 | 0.7547 &pm; 0.0043     | 0.6038 &pm; 0.0108     | 0.7597 &pm; 0.0021     | 0.6109 &pm; 0.0107     |
| Schnet    | 4.0.0   |       50 | 0.7583 &pm; 0.0076     | 0.6119 &pm; 0.0159     | 0.7623 &pm; 0.0072     | 0.6191 &pm; 0.0105     |

#### Tox21MolNetDataset

Tox21 (MoleculeNet) consists of 7831 compounds as smiles and 12 different targets relevant to drug toxicity. We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | BACC                   | *Max. BACC*            | *Max. Accuracy*        | *Max. AUC*             |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |       50 | **0.9272 &pm; 0.0024** | **0.8321 &pm; 0.0103** | **0.6995 &pm; 0.0130** | **0.7123 &pm; 0.0142** | **0.9292 &pm; 0.0016** | **0.8417 &pm; 0.0075** |
| GAT       | 4.0.0   |       50 | 0.9243 &pm; 0.0022     | 0.8279 &pm; 0.0092     | 0.6504 &pm; 0.0074     | 0.6528 &pm; 0.0071     | 0.9246 &pm; 0.0021     | 0.8293 &pm; 0.0093     |
| GATv2     | 4.0.0   |       50 | 0.9222 &pm; 0.0019     | 0.8251 &pm; 0.0069     | 0.6760 &pm; 0.0140     | 0.6782 &pm; 0.0156     | 0.9246 &pm; 0.0025     | 0.8314 &pm; 0.0116     |
| GIN       | 4.0.0   |       50 | 0.9220 &pm; 0.0024     | 0.7986 &pm; 0.0180     | 0.6741 &pm; 0.0143     | 0.6882 &pm; 0.0151     | 0.9259 &pm; 0.0022     | 0.8248 &pm; 0.0130     |
| GraphSAGE | 4.0.0   |      100 | 0.9180 &pm; 0.0027     | 0.7976 &pm; 0.0087     | 0.6755 &pm; 0.0047     | 0.7083 &pm; 0.0114     | 0.9252 &pm; 0.0015     | 0.8225 &pm; 0.0142     |
| Schnet    | 4.0.0   |       50 | 0.9128 &pm; 0.0030     | 0.7719 &pm; 0.0139     | 0.6639 &pm; 0.0162     | 0.6820 &pm; 0.0115     | 0.9215 &pm; 0.0027     | 0.7980 &pm; 0.0079     |

