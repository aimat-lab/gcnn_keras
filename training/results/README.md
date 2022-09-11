# Summary of Benchmark Training

Note that these are the results for models within `kgcnn` implementation, and that training is not always done with optimal hyperparameter or splits, when comparing with literature.
This table is generated automatically from keras history logs.
Model weights and training statistics plots are not uploaded on github due to their file size.

## CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse attributes and 7 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   |
|:----------|:--------|---------:|:-----------------------|
| GAT       | 2.1.0   |      250 | **0.8667 &pm; 0.0069** |
| GATv2     | 2.1.0   |      250 | 0.8379 &pm; 0.0158     |
| GCN       | 2.1.0   |      300 | 0.8047 &pm; 0.0113     |
| GIN       | 2.1.0   |      500 | 0.8427 &pm; 0.0165     |
| GraphSAGE | 2.1.0   |      500 | 0.8486 &pm; 0.0097     |

## CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   |
|:----------|:--------|---------:|:-----------------------|
| GAT       | 2.1.0   |      250 | **0.6765 &pm; 0.0069** |
| GATv2     | 2.1.0   |     1000 | 0.6167 &pm; 0.0081     |
| GCN       | 2.1.0   |      300 | 0.6156 &pm; 0.0052     |
| GIN       | 2.1.0   |      800 | 0.6368 &pm; 0.0077     |
| GraphSAGE | 2.1.0   |      600 | 0.6145 &pm; 0.0073     |

## ESOLDataset

ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation. 

| model               | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:--------------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP         | 2.1.0   |      200 | 0.4359 &pm; 0.0295     | **0.5920 &pm; 0.0307** |
| CMPNN               | 2.1.0   |      600 | 0.4740 &pm; 0.0259     | 0.6766 &pm; 0.0266     |
| DimeNetPP           | 2.1.0   |      872 | 0.4572 &pm; 0.0304     | 0.6377 &pm; 0.0501     |
| DMPNN               | 2.1.0   |      300 | 0.4381 &pm; 0.0203     | 0.6321 &pm; 0.0478     |
| GAT                 | 2.1.0   |      500 | 0.4699 &pm; 0.0435     | 0.6711 &pm; 0.0745     |
| GATv2               | 2.1.0   |      500 | 0.4628 &pm; 0.0432     | 0.6615 &pm; 0.0565     |
| GCN                 | 2.1.0   |      800 | 0.5639 &pm; 0.0102     | 0.7995 &pm; 0.0324     |
| GIN                 | 2.1.0   |      300 | 0.5107 &pm; 0.0395     | 0.7241 &pm; 0.0441     |
| GIN.make_model_edge | 2.1.0   |      300 | 0.4761 &pm; 0.0259     | 0.6733 &pm; 0.0407     |
| GraphSAGE           | 2.1.0   |      500 | 0.4654 &pm; 0.0377     | 0.6556 &pm; 0.0697     |
| HamNet              | 2.1.0   |      400 | 0.5492 &pm; 0.0509     | 0.7645 &pm; 0.0676     |
| INorp               | 2.1.0   |      500 | 0.4828 &pm; 0.0201     | 0.6748 &pm; 0.0350     |
| Megnet              | 2.1.0   |      800 | 0.5597 &pm; 0.0314     | 0.7972 &pm; 0.0439     |
| NMPN                | 2.1.0   |      800 | 0.5706 &pm; 0.0497     | 0.8144 &pm; 0.0710     |
| PAiNN               | 2.1.0   |      250 | **0.4182 &pm; 0.0198** | 0.5961 &pm; 0.0344     |
| Schnet              | 2.1.0   |      800 | 0.4682 &pm; 0.0272     | 0.6539 &pm; 0.0471     |

## LipopDataset

Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles. Graph labels for regression are octanol/water distribution coefficient (logD at pH 7.4). We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.4644 &pm; 0.0245     | 0.6393 &pm; 0.0408     |
| CMPNN       | 2.1.0   |      600 | 0.4131 &pm; 0.0061     | 0.5835 &pm; 0.0094     |
| DMPNN       | 2.1.0   |      300 | **0.3781 &pm; 0.0091** | **0.5440 &pm; 0.0162** |
| GAT         | 2.1.0   |      500 | 0.5034 &pm; 0.0060     | 0.7037 &pm; 0.0202     |
| GATv2       | 2.1.0   |      500 | 0.3971 &pm; 0.0238     | 0.5688 &pm; 0.0609     |
| GIN         | 2.1.0   |      300 | 0.4503 &pm; 0.0106     | 0.6175 &pm; 0.0210     |
| HamNet      | 2.1.0   |      400 | 0.4535 &pm; 0.0119     | 0.6305 &pm; 0.0244     |
| INorp       | 2.1.0   |      500 | 0.4668 &pm; 0.0118     | 0.6576 &pm; 0.0214     |
| PAiNN       | 2.1.0   |      250 | 0.4050 &pm; 0.0070     | 0.5837 &pm; 0.0162     |
| Schnet      | 2.1.0   |      800 | 0.4879 &pm; 0.0205     | 0.6535 &pm; 0.0320     |

## MatProjectEFormDataset

Materials Project dataset from Matbench with 132752 crystal structures and their corresponding formation energy in [eV/atom]. We use a random 10-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV/atom]          | RMSE [eV/atom]         |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.1.0   |     1000 | 0.0354 &pm; 0.0006     | 0.0847 &pm; 0.0037     |
| DimeNetPP.make_crystal_model | 2.1.0   |      750 | 0.0220 &pm; 0.0006     | 0.0623 &pm; 0.0036     |
| Megnet.make_crystal_model    | 2.1.0   |     1000 | 0.0239 &pm; 0.0005     | 0.0634 &pm; 0.0039     |
| PAiNN.make_crystal_model     | 2.1.0   |      800 | 0.0278 &pm; 0.0005     | 0.0676 &pm; 0.0038     |
| Schnet.make_crystal_model    | 2.1.0   |      800 | **0.0209 &pm; 0.0004** | **0.0514 &pm; 0.0028** |

## MutagenicityDataset

Mutagenicity dataset from TUDataset for classification with 4337 graphs. The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.7466 &pm; 0.0216     | 0.8274 &pm; 0.0187     |
| CMPNN       | 2.1.0   |      600 | 0.8098 &pm; 0.0068     | 0.8331 &pm; 0.0070     |
| DMPNN       | 2.1.0   |      300 | **0.8271 &pm; 0.0069** | 0.8685 &pm; 0.0133     |
| GAT         | 2.1.0   |      500 | 0.7902 &pm; 0.0125     | 0.8469 &pm; 0.0117     |
| GATv2       | 2.1.0   |      500 | 0.8084 &pm; 0.0130     | 0.8320 &pm; 0.0116     |
| GIN         | 2.1.0   |      300 | 0.8262 &pm; 0.0110     | **0.8818 &pm; 0.0045** |
| GraphSAGE   | 2.1.0   |      500 | 0.8063 &pm; 0.0097     | 0.8449 &pm; 0.0147     |
| INorp       | 2.1.0   |      500 | 0.8040 &pm; 0.0113     | 0.8290 &pm; 0.0117     |

## MUTAGDataset

MUTAG dataset from TUDataset for classification with 188 graphs. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.8455 &pm; 0.0600     | 0.8893 &pm; 0.0812     |
| CMPNN       | 2.1.0   |      600 | 0.8138 &pm; 0.0612     | 0.8133 &pm; 0.0680     |
| DMPNN       | 2.1.0   |      300 | 0.8506 &pm; 0.0447     | **0.9038 &pm; 0.0435** |
| GAT         | 2.1.0   |      500 | 0.8141 &pm; 0.0405     | 0.8698 &pm; 0.0499     |
| GATv2       | 2.1.0   |      500 | 0.7660 &pm; 0.0303     | 0.7885 &pm; 0.0433     |
| GIN         | 2.1.0   |      300 | 0.8243 &pm; 0.0372     | 0.8570 &pm; 0.0422     |
| GraphSAGE   | 2.1.0   |      500 | **0.8512 &pm; 0.0263** | 0.8707 &pm; 0.0449     |
| INorp       | 2.1.0   |      500 | 0.8450 &pm; 0.0682     | 0.8519 &pm; 0.1071     |

## FreeSolvDataset

FreeSolv (MoleculeNet) consists of 642 compounds as smiles and their corresponding hydration free energy for small neutral molecules in water. We use a random 5-fold cross-validation. *Min. MAE/RMSE* denotes the smallest test MAE/RMSE observed for any epoch during training. 

| model               | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       | *Min. MAE*             | *Min. RMSE*            |
|:--------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP         | 2.1.0   |      200 | 0.6055 &pm; 0.0654     | 0.9643 &pm; 0.1413     | 0.5254 &pm; 0.0506     | 0.8573 &pm; 0.1235     |
| CMPNN               | 2.1.0   |      600 | 0.5319 &pm; 0.0655     | 0.9262 &pm; 0.1597     | 0.4983 &pm; 0.0589     | 0.8673 &pm; 0.1571     |
| DimeNetPP           | 2.1.0   |      300 | 0.5791 &pm; 0.0649     | 0.9439 &pm; 0.1602     | 0.5519 &pm; 0.0648     | 0.8618 &pm; 0.0973     |
| DMPNN               | 2.1.0   |      300 | 0.5305 &pm; 0.0474     | **0.9070 &pm; 0.1497** | **0.4809 &pm; 0.0450** | **0.8192 &pm; 0.1484** |
| GAT                 | 2.1.0   |      500 | 0.6401 &pm; 0.0892     | 1.0845 &pm; 0.2040     | 0.6237 &pm; 0.0841     | 1.0494 &pm; 0.1967     |
| GATv2               | 2.1.0   |      500 | 0.6390 &pm; 0.0467     | 1.1203 &pm; 0.1491     | 0.5988 &pm; 0.0355     | 0.9891 &pm; 0.1020     |
| GCN                 | 2.1.0   |      800 | 0.7766 &pm; 0.0774     | 1.3245 &pm; 0.2008     | 0.7176 &pm; 0.0542     | 1.1710 &pm; 0.0881     |
| GIN                 | 2.1.0   |      300 | 0.7112 &pm; 0.0917     | 1.1421 &pm; 0.1469     | 0.6318 &pm; 0.0531     | 1.0502 &pm; 0.1093     |
| GIN.make_model_edge | 2.1.0   |      300 | 0.6197 &pm; 0.0685     | 1.0584 &pm; 0.1756     | 0.5816 &pm; 0.0809     | 0.9840 &pm; 0.1753     |
| GraphSAGE           | 2.1.0   |      500 | 0.5667 &pm; 0.0577     | 0.9861 &pm; 0.1328     | 0.5496 &pm; 0.0575     | 0.9236 &pm; 0.1444     |
| HamNet              | 2.1.0   |      400 | 0.6395 &pm; 0.0496     | 1.0508 &pm; 0.0827     | 0.5862 &pm; 0.0446     | 0.9691 &pm; 0.0884     |
| INorp               | 2.1.0   |      500 | 0.6448 &pm; 0.0607     | 1.0911 &pm; 0.1530     | 0.6021 &pm; 0.0640     | 0.9915 &pm; 0.1678     |
| Megnet              | 2.1.0   |      800 | 0.9749 &pm; 0.0429     | 1.5328 &pm; 0.0862     | 0.8850 &pm; 0.0481     | 1.3589 &pm; 0.0661     |
| NMPN                | 2.1.0   |      800 | 0.6393 &pm; 0.0808     | 1.0830 &pm; 0.1283     | 0.5886 &pm; 0.0663     | 0.9532 &pm; 0.1101     |
| PAiNN               | 2.1.0   |      250 | **0.5128 &pm; 0.0565** | 0.9403 &pm; 0.1387     | 0.4854 &pm; 0.0474     | 0.8569 &pm; 0.1270     |
| Schnet              | 2.1.0   |      800 | 0.5980 &pm; 0.0556     | 1.0614 &pm; 0.1531     | 0.5616 &pm; 0.0456     | 0.9441 &pm; 0.1021     |

## PROTEINSDataset

TUDataset of proteins that are classified as enzymes or non-enzymes. Nodes represent the amino acids of the protein. We use random 5-fold cross-validation. *Max. Accuracy/AUC* denotes the highest test Accuracy/AUC observed for any epoch during training. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.7269 &pm; 0.0280     | 0.7925 &pm; 0.0250     | 0.7772 &pm; 0.0270     | **0.8361 &pm; 0.0278** |
| CMPNN       | 2.1.0   |      600 | 0.7377 &pm; 0.0164     | 0.7532 &pm; 0.0174     | 0.7763 &pm; 0.0210     | 0.8337 &pm; 0.0252     |
| DMPNN       | 2.1.0   |      300 | **0.7395 &pm; 0.0300** | **0.8038 &pm; 0.0365** | **0.7907 &pm; 0.0272** | 0.8357 &pm; 0.0274     |
| GAT         | 2.1.0   |      500 | 0.7368 &pm; 0.0330     | 0.7991 &pm; 0.0303     | 0.7817 &pm; 0.0358     | 0.8278 &pm; 0.0367     |
| GATv2       | 2.1.0   |      500 | 0.6999 &pm; 0.0266     | 0.7137 &pm; 0.0177     | 0.7835 &pm; 0.0348     | 0.8270 &pm; 0.0387     |
| GIN         | 2.1.0   |      150 | 0.7098 &pm; 0.0357     | 0.7437 &pm; 0.0454     | 0.7449 &pm; 0.0325     | 0.7728 &pm; 0.0401     |
| GraphSAGE   | 2.1.0   |      500 | 0.6937 &pm; 0.0273     | 0.7263 &pm; 0.0391     | 0.7718 &pm; 0.0374     | 0.8011 &pm; 0.0325     |
| INorp       | 2.1.0   |      500 | 0.7242 &pm; 0.0359     | 0.7333 &pm; 0.0228     | 0.7763 &pm; 0.0295     | 0.8178 &pm; 0.0327     |

## Tox21MolNetDataset

Tox21 (MoleculeNet) consists of 7831 compounds as smiles and 12 different targets relevant to drug toxicity. We use random 5-fold cross-validation. *Max. Accuracy/AUC* denotes the highest test Accuracy/AUC observed for any epoch during training. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |       50 | 0.9359 &pm; 0.0023     | 0.8166 &pm; 0.0062     | 0.9402 &pm; 0.0016     | 0.8435 &pm; 0.0066     |
| CMPNN       | 2.1.0   |       30 | 0.9301 &pm; 0.0035     | 0.7478 &pm; 0.0695     | 0.9320 &pm; 0.0039     | 0.7566 &pm; 0.0773     |
| DMPNN       | 2.1.0   |       50 | **0.9396 &pm; 0.0031** | **0.8348 &pm; 0.0065** | **0.9414 &pm; 0.0020** | **0.8439 &pm; 0.0050** |
| GAT         | 2.1.0   |       50 | 0.9354 &pm; 0.0032     | 0.8326 &pm; 0.0104     | 0.9368 &pm; 0.0024     | 0.8349 &pm; 0.0089     |
| GATv2       | 2.1.0   |       50 | 0.9360 &pm; 0.0019     | 0.8295 &pm; 0.0070     | 0.9378 &pm; 0.0024     | 0.8351 &pm; 0.0061     |
| GIN         | 2.1.0   |       50 | 0.9368 &pm; 0.0021     | 0.8298 &pm; 0.0088     | 0.9395 &pm; 0.0019     | 0.8396 &pm; 0.0049     |
| GraphSAGE   | 2.1.0   |      100 | 0.9313 &pm; 0.0041     | 0.8042 &pm; 0.0090     | 0.9388 &pm; 0.0029     | 0.8361 &pm; 0.0085     |
| INorp       | 2.1.0   |       50 | 0.9350 &pm; 0.0032     | 0.8146 &pm; 0.0072     | 0.9372 &pm; 0.0023     | 0.8269 &pm; 0.0068     |
| Schnet      | 2.1.0   |       50 | 0.9316 &pm; 0.0029     | 0.7875 &pm; 0.0102     | 0.9354 &pm; 0.0024     | 0.8107 &pm; 0.0037     |

## QM7Dataset

QM7 dataset is a subset of GDB-13. Molecules of up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. We use a random 5-fold cross-validation. The atomization energies are given in kcal/mol and are ranging from -800 to -2000 kcal/mol). 

| model   | kgcnn   |   epochs | MAE [kcal/mol]         | RMSE [kcal/mol]        |
|:--------|:--------|---------:|:-----------------------|:-----------------------|
| Megnet  | 2.1.0   |      800 | **1.4922 &pm; 0.0680** | **2.8795 &pm; 0.5466** |
| Schnet  | 2.1.0   |      800 | 2.5247 &pm; 0.2063     | 8.8590 &pm; 4.8022     |

## QM9Dataset

QM9 dataset of 134k stable small organic molecules made up of C,H,O,N,F. Labels include geometric, energetic, electronic, and thermodynamic properties. We use a random 10-fold cross-validation. Test errors are MAE and for energies are given in [eV]. 

| model   | kgcnn   |   epochs | HOMO                   |
|:--------|:--------|---------:|:-----------------------|
| Megnet  | 2.1.0   |      800 | 0.0423 &pm; 0.0014     |
| Schnet  | 2.1.0   |      800 | **0.0351 &pm; 0.0005** |

