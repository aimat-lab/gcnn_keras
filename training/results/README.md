# Summary of Benchmark Training

Note that these are the results for models within `kgcnn` implementation, and that training is not always done with optimal hyperparameter or splits, when comparing with literature.
This table is generated automatically from keras history logs.
Model weights and training statistics plots are not uploaded on github due to their file size.

*Max.* or *Min.* denotes the best test error observed for any epoch during training.
To show overall best test error run ``python3 summary.py --min_max True``.
If not noted otherwise, we use a (fixed) random k-fold split for validation errors.

## CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse attributes and 7 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   |
|:----------|:--------|---------:|:-----------------------|
| GAT       | 2.1.0   |      250 | 0.8490 &pm; 0.0122     |
| GATv2     | 2.1.0   |      250 | 0.8261 &pm; 0.0106     |
| GCN       | 2.1.0   |      300 | 0.8076 &pm; 0.0119     |
| GIN       | 2.1.0   |      500 | 0.8058 &pm; 0.0449     |
| GraphSAGE | 2.1.0   |      500 | **0.8512 &pm; 0.0100** |

## CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   |
|:----------|:--------|---------:|:-----------------------|
| GAT       | 2.1.0   |      250 | 0.6147 &pm; 0.0077     |
| GATv2     | 2.1.0   |     1000 | 0.6144 &pm; 0.0110     |
| GCN       | 2.1.0   |      300 | 0.6136 &pm; 0.0057     |
| GIN       | 2.1.0   |      800 | **0.6347 &pm; 0.0117** |
| GraphSAGE | 2.1.0   |      600 | 0.6133 &pm; 0.0045     |

## ESOLDataset

ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation. 

| model               | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:--------------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP         | 2.1.0   |      200 | 0.4562 &pm; 0.0084     | 0.6322 &pm; 0.0257     |
| CMPNN               | 2.1.0   |      600 | 0.4814 &pm; 0.0265     | 0.6821 &pm; 0.0193     |
| DimeNetPP           | 2.1.0   |      872 | 0.4576 &pm; 0.0422     | 0.6505 &pm; 0.0708     |
| DMPNN               | 2.1.0   |      300 | 0.4476 &pm; 0.0165     | 0.6349 &pm; 0.0152     |
| GAT                 | 2.1.0   |      500 | 0.4857 &pm; 0.0239     | 0.7028 &pm; 0.0356     |
| GATv2               | 2.1.0   |      500 | 0.4691 &pm; 0.0262     | 0.6724 &pm; 0.0348     |
| GCN                 | 2.1.0   |      800 | 0.5917 &pm; 0.0301     | 0.8118 &pm; 0.0465     |
| GIN                 | 2.1.0   |      300 | 0.5023 &pm; 0.0182     | 0.6997 &pm; 0.0300     |
| GIN.make_model_edge | 2.1.0   |      300 | 0.4881 &pm; 0.0173     | 0.6759 &pm; 0.0229     |
| GraphSAGE           | 2.1.0   |      500 | 0.5003 &pm; 0.0445     | 0.7242 &pm; 0.0791     |
| HamNet              | 2.1.0   |      400 | 0.5485 &pm; 0.0225     | 0.7605 &pm; 0.0210     |
| INorp               | 2.1.0   |      500 | 0.4856 &pm; 0.0145     | 0.6801 &pm; 0.0252     |
| Megnet              | 2.1.0   |      800 | 0.5446 &pm; 0.0142     | 0.7651 &pm; 0.0410     |
| NMPN                | 2.1.0   |      800 | 0.5820 &pm; 0.0451     | 0.8231 &pm; 0.0682     |
| PAiNN               | 2.1.0   |      250 | **0.4291 &pm; 0.0164** | **0.6014 &pm; 0.0238** |
| Schnet              | 2.1.0   |      800 | 0.4579 &pm; 0.0259     | 0.6527 &pm; 0.0411     |

## LipopDataset

Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles. Graph labels for regression are octanol/water distribution coefficient (logD at pH 7.4). We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.4511 &pm; 0.0104     | 0.6193 &pm; 0.0149     |
| CMPNN       | 2.1.0   |      600 | 0.4129 &pm; 0.0069     | 0.5752 &pm; 0.0094     |
| DMPNN       | 2.1.0   |      300 | **0.3809 &pm; 0.0137** | **0.5503 &pm; 0.0251** |
| GAT         | 2.1.0   |      500 | 0.4954 &pm; 0.0172     | 0.6962 &pm; 0.0351     |
| GATv2       | 2.1.0   |      500 | 0.4081 &pm; 0.0099     | 0.5876 &pm; 0.0128     |
| GIN         | 2.1.0   |      300 | 0.4528 &pm; 0.0069     | 0.6382 &pm; 0.0286     |
| HamNet      | 2.1.0   |      400 | 0.4546 &pm; 0.0042     | 0.6293 &pm; 0.0139     |
| INorp       | 2.1.0   |      500 | 0.4635 &pm; 0.0106     | 0.6529 &pm; 0.0141     |
| PAiNN       | 2.1.0   |      250 | 0.4033 &pm; 0.0123     | 0.5798 &pm; 0.0281     |
| Schnet      | 2.1.0   |      800 | 0.4788 &pm; 0.0046     | 0.6450 &pm; 0.0036     |

## MutagenicityDataset

Mutagenicity dataset from TUDataset for classification with 4337 graphs. The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.7397 &pm; 0.0111     | 0.8207 &pm; 0.0111     |
| CMPNN       | 2.1.0   |      600 | 0.8102 &pm; 0.0157     | 0.8348 &pm; 0.0237     |
| DMPNN       | 2.1.0   |      300 | **0.8296 &pm; 0.0126** | 0.8714 &pm; 0.0075     |
| GAT         | 2.1.0   |      500 | 0.8008 &pm; 0.0115     | 0.8294 &pm; 0.0113     |
| GATv2       | 2.1.0   |      500 | 0.8029 &pm; 0.0122     | 0.8337 &pm; 0.0046     |
| GIN         | 2.1.0   |      300 | 0.8185 &pm; 0.0127     | **0.8734 &pm; 0.0094** |
| GraphSAGE   | 2.1.0   |      500 | 0.8165 &pm; 0.0061     | 0.8530 &pm; 0.0089     |
| INorp       | 2.1.0   |      500 | 0.7955 &pm; 0.0037     | 0.8255 &pm; 0.0047     |

## MUTAGDataset

MUTAG dataset from TUDataset for classification with 188 graphs. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.8085 &pm; 0.1031     | 0.8471 &pm; 0.0890     |
| CMPNN       | 2.1.0   |      600 | 0.7873 &pm; 0.0724     | 0.7811 &pm; 0.0762     |
| DMPNN       | 2.1.0   |      300 | 0.8461 &pm; 0.0474     | 0.8686 &pm; 0.0480     |
| GAT         | 2.1.0   |      500 | 0.8351 &pm; 0.0920     | 0.8779 &pm; 0.0854     |
| GATv2       | 2.1.0   |      500 | 0.8144 &pm; 0.0757     | 0.8400 &pm; 0.0688     |
| GIN         | 2.1.0   |      300 | **0.8512 &pm; 0.0485** | **0.8861 &pm; 0.0922** |
| GraphSAGE   | 2.1.0   |      500 | 0.8193 &pm; 0.0445     | 0.8560 &pm; 0.0651     |
| INorp       | 2.1.0   |      500 | 0.8407 &pm; 0.0829     | 0.8549 &pm; 0.0705     |

## FreeSolvDataset

FreeSolv (MoleculeNet) consists of 642 compounds as smiles and their corresponding hydration free energy for small neutral molecules in water. We use a random 5-fold cross-validation. 

| model               | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:--------------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP         | 2.1.0   |      200 | 0.5853 &pm; 0.0519     | 1.0168 &pm; 0.1386     |
| CMPNN               | 2.1.0   |      600 | 0.5319 &pm; 0.0655     | 0.9262 &pm; 0.1597     |
| DimeNetPP           | 2.1.0   |      300 | 0.5791 &pm; 0.0649     | 0.9439 &pm; 0.1602     |
| DMPNN               | 2.1.0   |      300 | 0.5305 &pm; 0.0474     | **0.9070 &pm; 0.1497** |
| GAT                 | 2.1.0   |      500 | 0.5970 &pm; 0.0776     | 1.0107 &pm; 0.1554     |
| GATv2               | 2.1.0   |      500 | 0.6390 &pm; 0.0467     | 1.1203 &pm; 0.1491     |
| GCN                 | 2.1.0   |      800 | 0.7766 &pm; 0.0774     | 1.3245 &pm; 0.2008     |
| GIN                 | 2.1.0   |      300 | 0.7161 &pm; 0.0492     | 1.1171 &pm; 0.1233     |
| GIN.make_model_edge | 2.1.0   |      300 | 0.6285 &pm; 0.0588     | 1.0457 &pm; 0.1458     |
| GraphSAGE           | 2.1.0   |      500 | 0.5667 &pm; 0.0577     | 0.9861 &pm; 0.1328     |
| HamNet              | 2.1.0   |      400 | 0.6395 &pm; 0.0496     | 1.0508 &pm; 0.0827     |
| INorp               | 2.1.0   |      500 | 0.6448 &pm; 0.0607     | 1.0911 &pm; 0.1530     |
| Megnet              | 2.1.0   |      800 | 0.9749 &pm; 0.0429     | 1.5328 &pm; 0.0862     |
| NMPN                | 2.1.0   |      800 | 0.6393 &pm; 0.0808     | 1.0830 &pm; 0.1283     |
| PAiNN               | 2.1.0   |      250 | **0.5128 &pm; 0.0565** | 0.9403 &pm; 0.1387     |
| Schnet              | 2.1.0   |      800 | 0.5980 &pm; 0.0556     | 1.0614 &pm; 0.1531     |

## PROTEINSDataset

TUDataset of proteins that are classified as enzymes or non-enzymes. Nodes represent the amino acids of the protein. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.7296 &pm; 0.0126     | 0.7967 &pm; 0.0264     |
| CMPNN       | 2.1.0   |      600 | 0.7377 &pm; 0.0164     | 0.7532 &pm; 0.0174     |
| DMPNN       | 2.1.0   |      300 | **0.7395 &pm; 0.0300** | **0.8038 &pm; 0.0365** |
| GAT         | 2.1.0   |      500 | 0.7314 &pm; 0.0283     | 0.7884 &pm; 0.0404     |
| GATv2       | 2.1.0   |      500 | 0.6999 &pm; 0.0266     | 0.7137 &pm; 0.0177     |
| GIN         | 2.1.0   |      150 | 0.7098 &pm; 0.0357     | 0.7437 &pm; 0.0454     |
| GraphSAGE   | 2.1.0   |      500 | 0.6937 &pm; 0.0273     | 0.7263 &pm; 0.0391     |
| INorp       | 2.1.0   |      500 | 0.7242 &pm; 0.0359     | 0.7333 &pm; 0.0228     |

## Tox21MolNetDataset

Tox21 (MoleculeNet) consists of 7831 compounds as smiles and 12 different targets relevant to drug toxicity. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |       50 | 0.9340 &pm; 0.0039     | 0.8134 &pm; 0.0062     |
| CMPNN       | 2.1.0   |       30 | 0.9301 &pm; 0.0035     | 0.7478 &pm; 0.0695     |
| DMPNN       | 2.1.0   |       50 | **0.9396 &pm; 0.0031** | **0.8348 &pm; 0.0065** |
| GAT         | 2.1.0   |       50 | 0.9358 &pm; 0.0023     | 0.8305 &pm; 0.0105     |
| GATv2       | 2.1.0   |       50 | 0.9360 &pm; 0.0019     | 0.8295 &pm; 0.0070     |
| GIN         | 2.1.0   |       50 | 0.9372 &pm; 0.0032     | 0.8291 &pm; 0.0069     |
| GraphSAGE   | 2.1.0   |      100 | 0.9313 &pm; 0.0041     | 0.8042 &pm; 0.0090     |
| INorp       | 2.1.0   |       50 | 0.9350 &pm; 0.0032     | 0.8146 &pm; 0.0072     |
| Schnet      | 2.1.0   |       50 | 0.9316 &pm; 0.0029     | 0.7875 &pm; 0.0102     |

## QM7Dataset

QM7 dataset is a subset of GDB-13. Molecules of up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. We use a random 5-fold cross-validation. The atomization energies are given in kcal/mol and are ranging from -800 to -2000 kcal/mol). 

| model   | kgcnn   |   epochs | MAE [kcal/mol]         | RMSE [kcal/mol]        |
|:--------|:--------|---------:|:-----------------------|:-----------------------|
| Megnet  | 2.1.0   |      800 | 1.4922 &pm; 0.0680     | **2.8795 &pm; 0.5466** |
| NMPN    | 2.1.0   |      500 | 6.5715 &pm; 0.4977     | 34.9752 &pm; 8.0740    |
| PAiNN   | 2.1.0   |      872 | **1.2690 &pm; 0.0938** | 4.5825 &pm; 2.2684     |
| Schnet  | 2.1.0   |      800 | 2.5247 &pm; 0.2063     | 8.8590 &pm; 4.8022     |

## QM9Dataset

QM9 dataset of 134k stable small organic molecules made up of C,H,O,N,F. Labels include geometric, energetic, electronic, and thermodynamic properties. We use a random 10-fold cross-validation. Test errors are MAE and for energies are given in [eV]. 

| model   | kgcnn   |   epochs | HOMO                   | LUMO               |
|:--------|:--------|---------:|:-----------------------|:-------------------|
| Megnet  | 2.1.0   |      800 | 0.0423 &pm; 0.0014     | **nan &pm; nan**   |
| NMPN    | 2.1.0   |      680 | 0.0627 &pm; 0.0013     | nan &pm; nan       |
| PAiNN   | 2.1.0   |      872 | **0.0287 &pm; 0.0068** | nan &pm; nan       |
| Schnet  | 2.1.0   |      800 | 0.0351 &pm; 0.0005     | 0.0293 &pm; 0.0006 |

