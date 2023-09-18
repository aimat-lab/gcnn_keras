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
| GAT       | 3.1.0   |      250 | 0.8442 &pm; 0.0137     |
| GATv2     | 3.1.0   |      250 | 0.8353 &pm; 0.0149     |
| GCN       | 3.1.0   |      300 | 0.8109 &pm; 0.0097     |
| GIN       | 3.1.0   |      500 | 0.8405 &pm; 0.0079     |
| GraphSAGE | 3.1.0   |      500 | **0.8497 &pm; 0.0102** |

#### CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   |
|:----------|:--------|---------:|:-----------------------|
| GAT       | 2.1.0   |      250 | 0.6147 &pm; 0.0077     |
| GATv2     | 2.1.0   |     1000 | 0.6144 &pm; 0.0110     |
| GCN       | 2.1.0   |      300 | 0.6136 &pm; 0.0057     |
| GIN       | 2.1.0   |      800 | **0.6347 &pm; 0.0117** |
| GraphSAGE | 2.1.0   |      600 | 0.6133 &pm; 0.0045     |

#### ESOLDataset

ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation. 

| model               | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       |
|:--------------------|:--------|---------:|:-----------------------|:-----------------------|
| AttentiveFP         | 3.0.0   |      200 | 0.4389 &pm; 0.0185     | 0.6103 &pm; 0.0256     |
| CMPNN               | 2.1.0   |      600 | 0.4814 &pm; 0.0265     | 0.6821 &pm; 0.0193     |
| DGIN                | 3.0.0   |      300 | 0.4311 &pm; 0.0243     | 0.6104 &pm; 0.0452     |
| DimeNetPP           | 2.1.0   |      872 | 0.4576 &pm; 0.0422     | 0.6505 &pm; 0.0708     |
| DMPNN               | 2.1.0   |      300 | 0.4476 &pm; 0.0165     | 0.6349 &pm; 0.0152     |
| GAT                 | 3.1.0   |      500 | 0.4736 &pm; 0.0124     | 0.6852 &pm; 0.0323     |
| GATv2               | 2.1.0   |      500 | 0.4691 &pm; 0.0262     | 0.6724 &pm; 0.0348     |
| GCN                 | 3.1.0   |      800 | 0.5797 &pm; 0.0326     | 0.8011 &pm; 0.0542     |
| GIN                 | 3.1.0   |      300 | 0.4942 &pm; 0.0097     | 0.6870 &pm; 0.0109     |
| GIN.make_model_edge | 2.1.0   |      300 | 0.4881 &pm; 0.0173     | 0.6759 &pm; 0.0229     |
| GNNFilm             | 2.2.0   |      800 | 0.5145 &pm; 0.0158     | 0.7166 &pm; 0.0342     |
| GraphSAGE           | 2.1.0   |      500 | 0.5003 &pm; 0.0445     | 0.7242 &pm; 0.0791     |
| HamNet              | 2.1.0   |      400 | 0.5485 &pm; 0.0225     | 0.7605 &pm; 0.0210     |
| HDNNP2nd            | 2.2.0   |      500 | 0.7085 &pm; 0.0830     | 0.9806 &pm; 0.1386     |
| INorp               | 2.1.0   |      500 | 0.4856 &pm; 0.0145     | 0.6801 &pm; 0.0252     |
| MAT                 | 2.1.1   |      400 | 0.5341 &pm; 0.0263     | 0.7232 &pm; 0.0448     |
| MEGAN               | 2.2.1   |      400 | 0.4305 &pm; 0.0072     | 0.6073 &pm; 0.0186     |
| Megnet              | 2.1.0   |      800 | 0.5446 &pm; 0.0142     | 0.7651 &pm; 0.0410     |
| NMPN                | 2.1.0   |      800 | 0.5045 &pm; 0.0217     | 0.7092 &pm; 0.0482     |
| PAiNN               | 2.1.0   |      250 | **0.4291 &pm; 0.0164** | **0.6014 &pm; 0.0238** |
| RGCN                | 2.2.0   |      800 | 0.5014 &pm; 0.0274     | 0.7028 &pm; 0.0332     |
| rGIN                | 3.0.0   |      300 | 0.6159 &pm; 0.0166     | 0.8445 &pm; 0.0147     |
| Schnet              | 3.1.0   |      800 | 0.4606 &pm; 0.0310     | 0.6588 &pm; 0.0578     |

#### LipopDataset

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
| MEGAN       | 2.1.0   |      400 | 0.3997 &pm; 0.0060     | 0.5635 &pm; 0.0114     |
| PAiNN       | 2.1.0   |      250 | 0.4033 &pm; 0.0123     | 0.5798 &pm; 0.0281     |
| Schnet      | 2.1.0   |      800 | 0.4788 &pm; 0.0046     | 0.6450 &pm; 0.0036     |

#### MatProjectJdft2dDataset

Materials Project dataset from Matbench with 636 crystal structures and their corresponding Exfoliation energy (meV/atom). We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [meV/atom]           | RMSE [meV/atom]           |
|:-----------------------------|:--------|---------:|:-------------------------|:--------------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 42.6352 &pm; 9.6715      | 112.4714 &pm; 37.9213     |
| coGN                         | 3.0.2   |      800 | **39.4277 &pm; 10.5046** | **111.8743 &pm; 39.3128** |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 49.2113 &pm; 12.7431     | 124.7198 &pm; 38.4492     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 56.5205 &pm; 10.8723     | 136.3116 &pm; 31.2617     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 50.5886 &pm; 9.9009      | 117.7118 &pm; 33.4786     |
| Schnet.make_crystal_model    | 3.1.0   |      800 | 49.3261 &pm; 9.9489      | 124.4529 &pm; 34.8170     |

#### MatProjectPhononsDataset

Materials Project dataset from Matbench with 1,265 crystal structures and their corresponding vibration properties in [1/cm]. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV/atom]           | RMSE [eV/atom]          |
|:-----------------------------|:--------|---------:|:------------------------|:------------------------|
| CGCNN.make_crystal_model     | 2.1.1   |     1000 | 46.1204 &pm; 3.2640     | 106.4514 &pm; 16.9401   |
| DimeNetPP.make_crystal_model | 2.1.1   |      780 | 36.7288 &pm; 1.3484     | 81.5038 &pm; 10.3550    |
| MEGAN                        | 2.1.1   |      400 | 50.3682 &pm; 7.2162     | 121.6629 &pm; 27.4599   |
| Megnet.make_crystal_model    | 2.1.0   |     1000 | **29.2085 &pm; 2.8130** | **53.9366 &pm; 7.0800** |
| NMPN.make_crystal_model      | 2.1.0   |      700 | 44.4253 &pm; 3.7905     | 91.1708 &pm; 23.8014    |
| PAiNN.make_crystal_model     | 2.1.1   |      800 | 47.2212 &pm; 3.8855     | 82.7834 &pm; 6.0730     |
| Schnet.make_crystal_model    | 2.2.3   |      800 | 40.2982 &pm; 1.6997     | 81.8959 &pm; 12.1697    |

#### MatProjectDielectricDataset

Materials Project dataset from Matbench with 4764 crystal structures and their corresponding Refractive index (unitless). We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [no unit]          | RMSE [no unit]         |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 0.3479 &pm; 0.0461     | 2.1384 &pm; 0.5135     |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 0.3337 &pm; 0.0608     | 1.8686 &pm; 0.6216     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 0.3485 &pm; 0.0443     | 2.0672 &pm; 0.5674     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.3587 &pm; 0.0518     | **1.8403 &pm; 0.6255** |
| Schnet.make_crystal_model    | 2.2.2   |      800 | **0.3241 &pm; 0.0375** | 2.0324 &pm; 0.5585     |

#### MatProjectLogGVRHDataset

Materials Project dataset from Matbench with 10987 crystal structures and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average shear moduli in GPa. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [log(GPa)]         | RMSE [log(GPa)]        |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 0.0847 &pm; 0.0020     | 0.1286 &pm; 0.0044     |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 0.0805 &pm; 0.0027     | 0.1259 &pm; 0.0056     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 0.0858 &pm; 0.0010     | 0.1337 &pm; 0.0035     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.0851 &pm; 0.0023     | 0.1284 &pm; 0.0057     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | **0.0798 &pm; 0.0011** | **0.1253 &pm; 0.0038** |

#### MatProjectLogKVRHDataset

Materials Project dataset from Matbench with 10987 crystal structures and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average bulk moduli in GPa. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [log(GPa)]         | RMSE [log(GPa)]        |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 0.0629 &pm; 0.0008     | 0.1198 &pm; 0.0037     |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | **0.0579 &pm; 0.0014** | **0.1120 &pm; 0.0045** |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 0.0660 &pm; 0.0020     | 0.1251 &pm; 0.0058     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.0646 &pm; 0.0015     | 0.1177 &pm; 0.0052     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | 0.0584 &pm; 0.0016     | 0.1143 &pm; 0.0062     |

#### MatProjectPerovskitesDataset

Materials Project dataset from Matbench with 18928 crystal structures and their corresponding Heat of formation of the entire 5-atom perovskite cell in eV. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV]               | RMSE [eV]              |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | **0.0316 &pm; 0.0012** | **0.0597 &pm; 0.0044** |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 0.0373 &pm; 0.0008     | 0.0660 &pm; 0.0038     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 0.0351 &pm; 0.0013     | 0.0636 &pm; 0.0025     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.0456 &pm; 0.0009     | 0.0742 &pm; 0.0024     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | 0.0347 &pm; 0.0007     | 0.0615 &pm; 0.0030     |

#### MatProjectGapDataset

Materials Project dataset from Matbench with 106113 crystal structures and their band gap as calculated by PBE DFT from the Materials Project, in eV. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV]               | RMSE [eV]              |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 0.2298 &pm; 0.0054     | 0.5394 &pm; 0.0102     |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 0.2089 &pm; 0.0022     | 0.4912 &pm; 0.0104     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | **0.2003 &pm; 0.0132** | **0.4839 &pm; 0.0303** |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.2220 &pm; 0.0037     | 0.5315 &pm; 0.0260     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | 0.9351 &pm; 0.3720     | 1.5027 &pm; 0.4929     |

#### MatProjectIsMetalDataset

Materials Project dataset from Matbench with 106113 crystal structures and their corresponding Metallicity determined with pymatgen. 1 if the compound is a metal, 0 if the compound is not a metal. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | Accuracy               | AUC                    |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |      100 | 0.8902 &pm; 0.0021     | 0.9380 &pm; 0.0013     |
| DimeNetPP.make_crystal_model | 2.2.2   |       78 | **0.9067 &pm; 0.0021** | 0.9463 &pm; 0.0013     |
| Megnet.make_crystal_model    | 2.2.2   |      100 | 0.9025 &pm; 0.0042     | **0.9559 &pm; 0.0027** |
| PAiNN.make_crystal_model     | 2.2.2   |       80 | 0.8941 &pm; 0.0029     | 0.9331 &pm; 0.0024     |
| Schnet.make_crystal_model    | 2.2.2   |       80 | 0.8937 &pm; 0.0045     | 0.9498 &pm; 0.0023     |

#### MatProjectEFormDataset

Materials Project dataset from Matbench with 132752 crystal structures and their corresponding formation energy in [eV/atom]. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV/atom]          | RMSE [eV/atom]         |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.1.1   |     1000 | 0.0369 &pm; 0.0003     | 0.0873 &pm; 0.0026     |
| coGN                         | 3.0.1   |      800 | **0.0169 &pm; 0.0002** | **0.0484 &pm; 0.0043** |
| DimeNetPP.make_crystal_model | 2.1.1   |      780 | 0.0233 &pm; 0.0005     | 0.0644 &pm; 0.0020     |
| MEGAN                        | 2.1.1   |      800 | 0.0397 &pm; 0.0009     | 0.0902 &pm; 0.0041     |
| Megnet.make_crystal_model    | 2.1.0   |     1000 | 0.0247 &pm; 0.0006     | 0.0639 &pm; 0.0028     |
| PAiNN.make_crystal_model     | 2.1.1   |      800 | 0.0244 &pm; 0.0002     | 0.0568 &pm; 0.0032     |
| Schnet.make_crystal_model    | 2.1.1   |      800 | 0.0215 &pm; 0.0003     | 0.0525 &pm; 0.0030     |

#### MutagenicityDataset

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
| MEGAN       | 2.1.1   |      500 | 0.8137 &pm; 0.0117     | 0.8591 &pm; 0.0077     |

#### MUTAGDataset

