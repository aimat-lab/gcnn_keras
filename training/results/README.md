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

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:----------|:--------|---------:|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |       50 | 0.9480 &pm; 0.0138     | 0.8297 &pm; 0.0568     |
| GAT       | 4.0.0   |       50 | **0.9480 &pm; 0.0070** | 0.8512 &pm; 0.0468     |
| GATv2     | 4.0.0   |       50 | 0.9372 &pm; 0.0155     | **0.8587 &pm; 0.0754** |
| GCN       | 4.0.0   |       50 | 0.9432 &pm; 0.0155     | 0.8555 &pm; 0.0593     |
| GIN       | 4.0.0   |       50 | 0.9412 &pm; 0.0034     | 0.8066 &pm; 0.0636     |
| GraphSAGE | 4.0.0   |      100 | 0.9412 &pm; 0.0073     | 0.8013 &pm; 0.0422     |
| Schnet    | 4.0.0   |       50 | 0.9277 &pm; 0.0102     | 0.6562 &pm; 0.0760     |

#### CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   |
|:----------|:--------|---------:|:-----------------------|
| DMPNN     | 4.0.0   |      300 | 0.2476 &pm; 0.1706     |
| GAT       | 4.0.0   |      250 | 0.6157 &pm; 0.0071     |
| GATv2     | 4.0.0   |     1000 | 0.6211 &pm; 0.0048     |
| GCN       | 4.0.0   |      300 | 0.6232 &pm; 0.0054     |
| GIN       | 4.0.0   |      800 | **0.6263 &pm; 0.0080** |
| GraphSAGE | 4.0.0   |      600 | 0.6151 &pm; 0.0053     |

#### CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse attributes and 7 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   |
|:----------|:--------|---------:|:-----------------------|
| DMPNN     | 4.0.0   |      300 | 0.8357 &pm; 0.0156     |
| GAT       | 4.0.0   |      250 | 0.8397 &pm; 0.0122     |
| GATv2     | 4.0.0   |      250 | 0.8331 &pm; 0.0104     |
| GCN       | 4.0.0   |      300 | 0.8072 &pm; 0.0109     |
| GIN       | 4.0.0   |      500 | 0.8279 &pm; 0.0170     |
| GraphSAGE | 4.0.0   |      500 | **0.8497 &pm; 0.0100** |

#### ESOLDataset

ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP | 4.0.0   |      200 | 0.4351 &pm; 0.0110     | 0.6080 &pm; 0.0207     |
| CMPNN       | 4.0.0   |      600 | 0.5276 &pm; 0.0154     | 0.7505 &pm; 0.0189     |
| DGIN        | 4.0.0   |      300 | 0.4434 &pm; 0.0252     | 0.6225 &pm; 0.0420     |
| DMPNN       | 4.0.0   |      300 | 0.4401 &pm; 0.0165     | 0.6203 &pm; 0.0292     |
| EGNN        | 4.0.0   |      800 | 0.4507 &pm; 0.0152     | 0.6563 &pm; 0.0370     |
| GAT         | 4.0.0   |      500 | 0.4818 &pm; 0.0240     | 0.6919 &pm; 0.0694     |
| GATv2       | 4.0.0   |      500 | 0.4598 &pm; 0.0234     | 0.6650 &pm; 0.0409     |
| GCN         | 4.0.0   |      800 | 0.4613 &pm; 0.0205     | 0.6534 &pm; 0.0513     |
| GIN         | 4.0.0   |      300 | 0.5369 &pm; 0.0334     | 0.7954 &pm; 0.0861     |
| GNNFilm     | 4.0.0   |      800 | 0.4854 &pm; 0.0368     | 0.6724 &pm; 0.0436     |
| GraphSAGE   | 4.0.0   |      500 | 0.4874 &pm; 0.0228     | 0.6982 &pm; 0.0608     |
| HDNNP2nd    | 4.0.0   |      500 | 0.7857 &pm; 0.0986     | 1.0467 &pm; 0.1367     |
| INorp       | 4.0.0   |      500 | 0.5055 &pm; 0.0436     | 0.7297 &pm; 0.0786     |
| MAT         | 4.0.0   |      400 | 0.5064 &pm; 0.0299     | 0.7194 &pm; 0.0630     |
| MEGAN       | 4.0.0   |      400 | **0.4281 &pm; 0.0201** | **0.6062 &pm; 0.0252** |
| Megnet      | 4.0.0   |      800 | 0.5679 &pm; 0.0310     | 0.8196 &pm; 0.0480     |
| MoGAT       | 4.0.0   |      200 | 0.4797 &pm; 0.0114     | 0.6533 &pm; 0.0114     |
| MXMNet      | 4.0.0   |      900 | 0.6486 &pm; 0.0633     | 1.0123 &pm; 0.2059     |
| NMPN        | 4.0.0   |      800 | 0.5046 &pm; 0.0266     | 0.7193 &pm; 0.0607     |
| PAiNN       | 4.0.0   |      250 | 0.4857 &pm; 0.0598     | 0.6650 &pm; 0.0674     |
| RGCN        | 4.0.0   |      800 | 0.4703 &pm; 0.0251     | 0.6529 &pm; 0.0318     |
| rGIN        | 4.0.0   |      300 | 0.5196 &pm; 0.0351     | 0.7142 &pm; 0.0263     |
| Schnet      | 4.0.0   |      800 | 0.4777 &pm; 0.0294     | 0.6977 &pm; 0.0538     |

#### FreeSolvDataset

FreeSolv (MoleculeNet) consists of 642 compounds as smiles and their corresponding hydration free energy for small neutral molecules in water. We use a random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:----------|:--------|---------:|:-----------------------|:-----------------------|
| CMPNN     | 4.0.0   |      600 | 0.5202 &pm; 0.0504     | 0.9339 &pm; 0.1286     |
| DGIN      | 4.0.0   |      300 | 0.5489 &pm; 0.0374     | 0.9448 &pm; 0.0787     |
| DimeNetPP | 4.0.0   |      872 | 0.6167 &pm; 0.0719     | 1.0302 &pm; 0.1717     |
| DMPNN     | 4.0.0   |      300 | 0.5487 &pm; 0.0754     | **0.9206 &pm; 0.1889** |
| EGNN      | 4.0.0   |      800 | 0.5386 &pm; 0.0548     | 1.0363 &pm; 0.1237     |
| GAT       | 4.0.0   |      500 | 0.6051 &pm; 0.0861     | 1.0326 &pm; 0.1819     |
| GATv2     | 4.0.0   |      500 | 0.6151 &pm; 0.0247     | 1.0535 &pm; 0.0817     |
| GCN       | 4.0.0   |      800 | 0.6400 &pm; 0.0834     | 1.0876 &pm; 0.1393     |
| GIN       | 4.0.0   |      300 | 0.8100 &pm; 0.1016     | 1.2695 &pm; 0.1192     |
| GNNFilm   | 4.0.0   |      800 | 0.6562 &pm; 0.0552     | 1.1597 &pm; 0.1245     |
| GraphSAGE | 4.0.0   |      500 | 0.5894 &pm; 0.0675     | 1.0009 &pm; 0.1491     |
| HamNet    | 4.0.0   |      400 | 0.6619 &pm; 0.0428     | 1.1410 &pm; 0.1120     |
| HDNNP2nd  | 4.0.0   |      500 | 1.0201 &pm; 0.1559     | 1.6351 &pm; 0.3419     |
| INorp     | 4.0.0   |      500 | 0.6612 &pm; 0.0188     | 1.1155 &pm; 0.1061     |
| MAT       | 4.0.0   |      400 | 0.8115 &pm; 0.0649     | 1.3099 &pm; 0.1235     |
| MEGAN     | 4.0.0   |      400 | 0.6303 &pm; 0.0550     | 1.0429 &pm; 0.1031     |
| Megnet    | 4.0.0   |      800 | 0.8878 &pm; 0.0528     | 1.4134 &pm; 0.1200     |
| MoGAT     | 4.0.0   |      200 | 0.7097 &pm; 0.0374     | 1.0911 &pm; 0.1334     |
| MXMNet    | 4.0.0   |      900 | 1.1386 &pm; 0.1979     | 3.0487 &pm; 2.1757     |
| RGCN      | 4.0.0   |      800 | **0.5128 &pm; 0.0810** | 0.9228 &pm; 0.1887     |
| rGIN      | 4.0.0   |      300 | 0.8503 &pm; 0.0613     | 1.3285 &pm; 0.0976     |
| Schnet    | 4.0.0   |      800 | 0.6070 &pm; 0.0285     | 1.0603 &pm; 0.0549     |

#### ISO17Dataset

The database consist of 129 molecules each containing 5,000 conformational geometries, energies and forces with a resolution of 1 femtosecond in the molecular dynamics trajectories. The molecules were randomly drawn from the largest set of isomers in the QM9 dataset. 

| model                   | kgcnn   |   epochs | Energy (test_within)   | Force (test_within)   |
|:------------------------|:--------|---------:|:-----------------------|:----------------------|
| Schnet.EnergyForceModel | 4.0.0   |     1000 | **0.0061 &pm; nan**    | **0.0134 &pm; nan**   |

#### LipopDataset

Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles. Graph labels for regression are octanol/water distribution coefficient (logD at pH 7.4). We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:----------|:--------|---------:|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |      300 | **0.3814 &pm; 0.0064** | **0.5462 &pm; 0.0095** |
| GAT       | 4.0.0   |      500 | 0.5168 &pm; 0.0088     | 0.7220 &pm; 0.0098     |
| GATv2     | 4.0.0   |      500 | 0.4342 &pm; 0.0104     | 0.6056 &pm; 0.0114     |
| GCN       | 4.0.0   |      800 | 0.4960 &pm; 0.0107     | 0.6833 &pm; 0.0155     |
| GIN       | 4.0.0   |      300 | 0.4745 &pm; 0.0101     | 0.6658 &pm; 0.0159     |
| GraphSAGE | 4.0.0   |      500 | 0.4333 &pm; 0.0217     | 0.6218 &pm; 0.0318     |
| Schnet    | 4.0.0   |      800 | 0.5657 &pm; 0.0202     | 0.7485 &pm; 0.0245     |

#### MD17Dataset

Energies and forces for molecular dynamics trajectories of eight organic molecules. All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. Errors are MAE for forces. Results are for the CCSD and CCSD(T) data in MD17. 

| model                   | kgcnn   |   epochs | Aspirin          | Toluene             | Malonaldehyde    | Benzene          | Ethanol             |
|:------------------------|:--------|---------:|:-----------------|:--------------------|:-----------------|:-----------------|:--------------------|
| PAiNN.EnergyForceModel  | 4.0.0   |     1000 | **nan &pm; nan** | **0.2815 &pm; nan** | **nan &pm; nan** | **nan &pm; nan** | 0.5805 &pm; nan     |
| Schnet.EnergyForceModel | 4.0.0   |     1000 | 1.2173 &pm; nan  | 0.7395 &pm; nan     | 0.8444 &pm; nan  | 0.3353 &pm; nan  | **0.4832 &pm; nan** |

#### MD17RevisedDataset

Energies and forces for molecular dynamics trajectories. All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. Errors are MAE for forces. 

| model                   | kgcnn   |   epochs | Aspirin                | Toluene                | Malonaldehyde          | Benzene                | Ethanol                |
|:------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| Schnet.EnergyForceModel | 4.0.0   |     1000 | **1.0389 &pm; 0.0071** | **0.5482 &pm; 0.0105** | **0.6727 &pm; 0.0132** | **0.2525 &pm; 0.0091** | **0.4471 &pm; 0.0199** |

#### MatProjectDielectricDataset

Materials Project dataset from Matbench with 4764 crystal structures and their corresponding Refractive index (unitless). We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [no unit]          | RMSE [no unit]         |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 0.3306 &pm; 0.0602     | 1.9736 &pm; 0.7324     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 0.3415 &pm; 0.0542     | 1.9637 &pm; 0.6323     |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 0.3362 &pm; 0.0550     | 2.0156 &pm; 0.5872     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 0.3289 &pm; 0.0489     | 1.8770 &pm; 0.6522     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 0.3539 &pm; 0.0433     | 1.8661 &pm; 0.5984     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | **0.3180 &pm; 0.0359** | **1.8509 &pm; 0.5854** |

#### MatProjectEFormDataset

Materials Project dataset from Matbench with 132752 crystal structures and their corresponding formation energy in [eV/atom]. We use a random 5-fold cross-validation. 

| model                     | kgcnn   |   epochs | MAE [eV/atom]          | RMSE [eV/atom]         |
|:--------------------------|:--------|---------:|:-----------------------|:-----------------------|
| Schnet.make_crystal_model | 4.0.0   |      800 | **0.0211 &pm; 0.0003** | **0.0510 &pm; 0.0024** |

#### MatProjectGapDataset

Materials Project dataset from Matbench with 106113 crystal structures and their band gap as calculated by PBE DFT from the Materials Project, in eV. We use a random 5-fold cross-validation. 

| model                     | kgcnn   |   epochs | MAE [eV]               | RMSE [eV]                 |
|:--------------------------|:--------|---------:|:-----------------------|:--------------------------|
| Schnet.make_crystal_model | 4.0.0   |      800 | **1.2226 &pm; 1.0573** | **58.3713 &pm; 114.2957** |

#### MatProjectIsMetalDataset

Materials Project dataset from Matbench with 106113 crystal structures and their corresponding Metallicity determined with pymatgen. 1 if the compound is a metal, 0 if the compound is not a metal. We use a random 5-fold cross-validation. 

| model                     | kgcnn   |   epochs | Accuracy               | AUC                    |
|:--------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model  | 4.0.0   |      100 | 0.8910 &pm; 0.0027     | 0.9406 &pm; 0.0024     |
| Schnet.make_crystal_model | 4.0.0   |       80 | **0.8953 &pm; 0.0058** | **0.9506 &pm; 0.0053** |

#### MatProjectJdft2dDataset

Materials Project dataset from Matbench with 636 crystal structures and their corresponding Exfoliation energy (meV/atom). We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [meV/atom]           | RMSE [meV/atom]           |
|:-----------------------------|:--------|---------:|:-------------------------|:--------------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 57.6974 &pm; 18.0803     | 140.6167 &pm; 44.8418     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 50.2880 &pm; 11.4199     | 126.0600 &pm; 38.3769     |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 51.1735 &pm; 9.1746      | 123.4178 &pm; 32.9582     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 59.3986 &pm; 10.9272     | 139.5943 &pm; 32.1129     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 49.3889 &pm; 11.5376     | 121.7087 &pm; 30.0472     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | **45.2412 &pm; 11.6395** | **115.6890 &pm; 39.0929** |

#### MatProjectLogGVRHDataset

Materials Project dataset from Matbench with 10987 crystal structures and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average shear moduli in GPa. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [log(GPa)]         | RMSE [log(GPa)]        |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 0.0874 &pm; 0.0022     | 0.1354 &pm; 0.0056     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 0.0839 &pm; 0.0027     | **0.1290 &pm; 0.0065** |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 0.0885 &pm; 0.0017     | 0.1360 &pm; 0.0054     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 0.0874 &pm; 0.0027     | 0.1324 &pm; 0.0045     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 0.0870 &pm; 0.0033     | 0.1332 &pm; 0.0103     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | **0.0836 &pm; 0.0021** | 0.1296 &pm; 0.0044     |

#### MatProjectLogKVRHDataset

Materials Project dataset from Matbench with 10987 crystal structures and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average bulk moduli in GPa. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [log(GPa)]         | RMSE [log(GPa)]        |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 0.0672 &pm; 0.0012     | 0.1265 &pm; 0.0042     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | **0.0604 &pm; 0.0023** | **0.1141 &pm; 0.0055** |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 0.0686 &pm; 0.0016     | 0.1285 &pm; 0.0061     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 0.0688 &pm; 0.0009     | 0.1262 &pm; 0.0031     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 0.0649 &pm; 0.0007     | 0.1170 &pm; 0.0048     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | 0.0635 &pm; 0.0016     | 0.1186 &pm; 0.0044     |

#### MatProjectPerovskitesDataset

Materials Project dataset from Matbench with 18928 crystal structures and their corresponding Heat of formation of the entire 5-atom perovskite cell in eV. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV]               | RMSE [eV]              |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 0.0425 &pm; 0.0011     | 0.0712 &pm; 0.0037     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 0.0447 &pm; 0.0016     | 0.0730 &pm; 0.0050     |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | 0.0388 &pm; 0.0017     | 0.0675 &pm; 0.0041     |
| NMPN.make_crystal_model      | 4.0.0   |      700 | **0.0381 &pm; 0.0009** | 0.0652 &pm; 0.0029     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 0.0474 &pm; 0.0003     | 0.0762 &pm; 0.0017     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | 0.0381 &pm; 0.0005     | **0.0645 &pm; 0.0024** |

#### MatProjectPhononsDataset

Materials Project dataset from Matbench with 1,265 crystal structures and their corresponding vibration properties in [1/cm]. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV/atom]           | RMSE [eV/atom]           |
|:-----------------------------|:--------|---------:|:------------------------|:-------------------------|
| CGCNN.make_crystal_model     | 4.0.0   |     1000 | 42.6447 &pm; 4.5721     | 92.1627 &pm; 21.4345     |
| DimeNetPP.make_crystal_model | 4.0.0   |      780 | 39.8893 &pm; 3.1280     | 77.5776 &pm; 16.0908     |
| Megnet.make_crystal_model    | 4.0.0   |     1000 | **30.6620 &pm; 2.9013** | **60.8733 &pm; 17.1448** |
| NMPN.make_crystal_model      | 4.0.0   |      700 | 45.9344 &pm; 5.7908     | 95.4136 &pm; 35.5401     |
| PAiNN.make_crystal_model     | 4.0.0   |      800 | 47.5408 &pm; 4.2815     | 86.6761 &pm; 11.9220     |
| Schnet.make_crystal_model    | 4.0.0   |      800 | 43.0692 &pm; 3.6227     | 88.5151 &pm; 20.0244     |

#### MUTAGDataset

MUTAG dataset from TUDataset for classification with 188 graphs. We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:----------|:--------|---------:|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |      300 | **0.8407 &pm; 0.0463** | 0.8567 &pm; 0.0511     |
| GAT       | 4.0.0   |      500 | 0.8141 &pm; 0.1077     | 0.8671 &pm; 0.0923     |
| GATv2     | 4.0.0   |      500 | 0.8193 &pm; 0.0945     | 0.8379 &pm; 0.1074     |
| GCN       | 4.0.0   |      800 | 0.7716 &pm; 0.0531     | 0.7956 &pm; 0.0909     |
| GIN       | 4.0.0   |      300 | 0.8091 &pm; 0.0781     | **0.8693 &pm; 0.0855** |
| GraphSAGE | 4.0.0   |      500 | 0.8357 &pm; 0.0798     | 0.8533 &pm; 0.0824     |

#### MutagenicityDataset

Mutagenicity dataset from TUDataset for classification with 4337 graphs. The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:----------|:--------|---------:|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |      300 | **0.8266 &pm; 0.0059** | **0.8708 &pm; 0.0076** |
| GAT       | 4.0.0   |      500 | 0.7989 &pm; 0.0114     | 0.8290 &pm; 0.0112     |
| GATv2     | 4.0.0   |      200 | 0.7674 &pm; 0.0048     | 0.8423 &pm; 0.0064     |
| GCN       | 4.0.0   |      800 | 0.7955 &pm; 0.0154     | 0.8191 &pm; 0.0137     |
| GIN       | 4.0.0   |      300 | 0.8118 &pm; 0.0091     | 0.8492 &pm; 0.0077     |
| GraphSAGE | 4.0.0   |      500 | 0.8195 &pm; 0.0126     | 0.8515 &pm; 0.0083     |

#### PROTEINSDataset

TUDataset of proteins that are classified as enzymes or non-enzymes. Nodes represent the amino acids of the protein. We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:----------|:--------|---------:|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |      300 | 0.7287 &pm; 0.0253     | **0.7970 &pm; 0.0343** |
| GAT       | 4.0.0   |      500 | **0.7314 &pm; 0.0357** | 0.7899 &pm; 0.0468     |
| GATv2     | 4.0.0   |      500 | 0.6720 &pm; 0.0595     | 0.6850 &pm; 0.0938     |
| GCN       | 4.0.0   |      800 | 0.7017 &pm; 0.0303     | 0.7211 &pm; 0.0254     |
| GIN       | 4.0.0   |      150 | 0.7224 &pm; 0.0343     | 0.7905 &pm; 0.0528     |
| GraphSAGE | 4.0.0   |      500 | 0.7009 &pm; 0.0398     | 0.7263 &pm; 0.0453     |

#### QM7Dataset

QM7 dataset is a subset of GDB-13. Molecules of up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. We use dataset-specific 5-fold cross-validation. The atomization energies are given in kcal/mol and are ranging from -800 to -2000 kcal/mol). 

| model     | kgcnn   |   epochs | MAE [kcal/mol]         | RMSE [kcal/mol]        |
|:----------|:--------|---------:|:-----------------------|:-----------------------|
| DimeNetPP | 4.0.0   |      872 | 3.4639 &pm; 0.2003     | 7.5327 &pm; 1.8190     |
| EGNN      | 4.0.0   |      800 | 1.7300 &pm; 0.1336     | 5.1268 &pm; 2.5134     |
| Megnet    | 4.0.0   |      800 | 1.5180 &pm; 0.0802     | 3.0321 &pm; 0.1936     |
| MXMNet    | 4.0.0   |      900 | **1.2431 &pm; 0.0820** | **2.6694 &pm; 0.2584** |
| NMPN      | 4.0.0   |      500 | 7.2907 &pm; 0.9061     | 38.1446 &pm; 12.1445   |
| PAiNN     | 4.0.0   |      872 | 1.5765 &pm; 0.0742     | 5.2705 &pm; 2.2848     |
| Schnet    | 4.0.0   |      800 | 3.4313 &pm; 0.4757     | 10.8978 &pm; 7.3863    |

#### QM9Dataset

QM9 dataset of 134k stable small organic molecules made up of C,H,O,N,F. Labels include geometric, energetic, electronic, and thermodynamic properties. We use a random 5-fold cross-validation, but not all splits are evaluated for cheaper evaluation. Test errors are MAE and for energies are given in [eV]. 

| model   | kgcnn   |   epochs | HOMO [eV]              | LUMO [eV]              | U0 [eV]                | H [eV]                 | G [eV]                 |
|:--------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| PAiNN   | 4.0.0   |      872 | 0.0483 &pm; 0.0275     | **0.0268 &pm; 0.0002** | **0.0099 &pm; 0.0003** | **0.0101 &pm; 0.0003** | **0.0110 &pm; 0.0002** |
| Schnet  | 4.0.0   |      800 | **0.0402 &pm; 0.0007** | 0.0340 &pm; 0.0001     | 0.0142 &pm; 0.0002     | 0.0146 &pm; 0.0002     | 0.0143 &pm; 0.0002     |

#### SIDERDataset

SIDER (MoleculeNet) consists of 1427 compounds as smiles and data for adverse drug reactions (ADR), grouped into 27 system organ classes. We use random 5-fold cross-validation.

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               |
|:----------|:--------|---------:|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |       50 | 0.7519 &pm; 0.0055     | **0.6280 &pm; 0.0173** |
| GAT       | 4.0.0   |       50 | **0.7595 &pm; 0.0034** | 0.6224 &pm; 0.0106     |
| GATv2     | 4.0.0   |       50 | 0.7548 &pm; 0.0052     | 0.6152 &pm; 0.0154     |
| GIN       | 4.0.0   |       50 | 0.7472 &pm; 0.0055     | 0.5995 &pm; 0.0058     |
| GraphSAGE | 4.0.0   |       30 | 0.7547 &pm; 0.0043     | 0.6038 &pm; 0.0108     |
| Schnet    | 4.0.0   |       50 | 0.7583 &pm; 0.0076     | 0.6119 &pm; 0.0159     |

#### Tox21MolNetDataset

Tox21 (MoleculeNet) consists of 7831 compounds as smiles and 12 different targets relevant to drug toxicity. We use random 5-fold cross-validation. 

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | BACC                   |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|
| DMPNN     | 4.0.0   |       50 | **0.9272 &pm; 0.0024** | **0.8321 &pm; 0.0103** | **0.6995 &pm; 0.0130** |
| GAT       | 4.0.0   |       50 | 0.9243 &pm; 0.0022     | 0.8279 &pm; 0.0092     | 0.6504 &pm; 0.0074     |
| GATv2     | 4.0.0   |       50 | 0.9222 &pm; 0.0019     | 0.8251 &pm; 0.0069     | 0.6760 &pm; 0.0140     |
| GIN       | 4.0.0   |       50 | 0.9220 &pm; 0.0024     | 0.7986 &pm; 0.0180     | 0.6741 &pm; 0.0143     |
| GraphSAGE | 4.0.0   |      100 | 0.9180 &pm; 0.0027     | 0.7976 &pm; 0.0087     | 0.6755 &pm; 0.0047     |
| Schnet    | 4.0.0   |       50 | 0.9128 &pm; 0.0030     | 0.7719 &pm; 0.0139     | 0.6639 &pm; 0.0162     |

