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

| model     | kgcnn   |   epochs | Categorical accuracy   | *Max. Categorical accuracy*   |
|:----------|:--------|---------:|:-----------------------|:------------------------------|
| GAT       | 2.1.0   |      250 | 0.8490 &pm; 0.0122     | 0.8645 &pm; 0.0072            |
| GATv2     | 2.1.0   |      250 | 0.8261 &pm; 0.0106     | 0.8427 &pm; 0.0124            |
| GCN       | 2.1.0   |      300 | 0.8076 &pm; 0.0119     | 0.8493 &pm; 0.0122            |
| GIN       | 2.1.0   |      500 | 0.8058 &pm; 0.0449     | 0.8320 &pm; 0.0252            |
| GraphSAGE | 2.1.0   |      500 | **0.8512 &pm; 0.0100** | **0.8652 &pm; 0.0065**        |

#### CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes. 

| model     | kgcnn   |   epochs | Categorical accuracy   | *Max. Categorical accuracy*   |
|:----------|:--------|---------:|:-----------------------|:------------------------------|
| GAT       | 2.1.0   |      250 | 0.6147 &pm; 0.0077     | 0.6302 &pm; 0.0069            |
| GATv2     | 2.1.0   |     1000 | 0.6144 &pm; 0.0110     | 0.6327 &pm; 0.0040            |
| GCN       | 2.1.0   |      300 | 0.6136 &pm; 0.0057     | 0.6206 &pm; 0.0071            |
| GIN       | 2.1.0   |      800 | **0.6347 &pm; 0.0117** | 0.6457 &pm; 0.0068            |
| GraphSAGE | 2.1.0   |      600 | 0.6133 &pm; 0.0045     | **0.6467 &pm; 0.0022**        |

#### ESOLDataset

ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation. 

| model               | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       | *Min. MAE*             | *Min. RMSE*            |
|:--------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP         | 3.0.0   |      200 | 0.4389 &pm; 0.0185     | 0.6103 &pm; 0.0256     | **0.4009 &pm; 0.0088** | **0.5590 &pm; 0.0190** |
| CMPNN               | 2.1.0   |      600 | 0.4814 &pm; 0.0265     | 0.6821 &pm; 0.0193     | 0.4596 &pm; 0.0155     | 0.6437 &pm; 0.0192     |
| DGIN                | 3.0.0   |      300 | 0.4311 &pm; 0.0243     | 0.6104 &pm; 0.0452     | 0.4206 &pm; 0.0229     | 0.5942 &pm; 0.0359     |
| DimeNetPP           | 2.1.0   |      872 | 0.4576 &pm; 0.0422     | 0.6505 &pm; 0.0708     | 0.4474 &pm; 0.0371     | 0.6344 &pm; 0.0642     |
| DMPNN               | 2.1.0   |      300 | 0.4476 &pm; 0.0165     | 0.6349 &pm; 0.0152     | 0.4227 &pm; 0.0137     | 0.5969 &pm; 0.0113     |
| GAT                 | 2.1.0   |      500 | 0.4857 &pm; 0.0239     | 0.7028 &pm; 0.0356     | 0.4624 &pm; 0.0184     | 0.6631 &pm; 0.0383     |
| GATv2               | 2.1.0   |      500 | 0.4691 &pm; 0.0262     | 0.6724 &pm; 0.0348     | 0.4313 &pm; 0.0163     | 0.6113 &pm; 0.0226     |
| GCN                 | 2.1.0   |      800 | 0.5917 &pm; 0.0301     | 0.8118 &pm; 0.0465     | 0.5656 &pm; 0.0200     | 0.7770 &pm; 0.0343     |
| GIN                 | 3.0.0   |      300 | 0.4892 &pm; 0.0182     | 0.6818 &pm; 0.0286     | 0.4809 &pm; 0.0195     | 0.6667 &pm; 0.0304     |
| GIN.make_model_edge | 2.1.0   |      300 | 0.4881 &pm; 0.0173     | 0.6759 &pm; 0.0229     | 0.4753 &pm; 0.0142     | 0.6546 &pm; 0.0226     |
| GNNFilm             | 2.2.0   |      800 | 0.5145 &pm; 0.0158     | 0.7166 &pm; 0.0342     | 0.5017 &pm; 0.0164     | 0.6966 &pm; 0.0347     |
| GraphSAGE           | 2.1.0   |      500 | 0.5003 &pm; 0.0445     | 0.7242 &pm; 0.0791     | 0.4850 &pm; 0.0361     | 0.6919 &pm; 0.0551     |
| HamNet              | 2.1.0   |      400 | 0.5485 &pm; 0.0225     | 0.7605 &pm; 0.0210     | 0.4875 &pm; 0.0180     | 0.6806 &pm; 0.0354     |
| HDNNP2nd            | 2.2.0   |      500 | 0.7085 &pm; 0.0830     | 0.9806 &pm; 0.1386     | 0.7076 &pm; 0.0830     | 0.9558 &pm; 0.1189     |
| INorp               | 2.1.0   |      500 | 0.4856 &pm; 0.0145     | 0.6801 &pm; 0.0252     | 0.4658 &pm; 0.0168     | 0.6505 &pm; 0.0171     |
| MAT                 | 2.1.1   |      400 | 0.5341 &pm; 0.0263     | 0.7232 &pm; 0.0448     | 0.5277 &pm; 0.0287     | 0.7184 &pm; 0.0474     |
| MEGAN               | 2.2.1   |      400 | 0.4305 &pm; 0.0072     | 0.6073 &pm; 0.0186     | 0.4184 &pm; 0.0087     | 0.5793 &pm; 0.0205     |
| Megnet              | 2.1.0   |      800 | 0.5446 &pm; 0.0142     | 0.7651 &pm; 0.0410     | 0.5051 &pm; 0.0174     | 0.7136 &pm; 0.0466     |
| NMPN                | 2.1.0   |      800 | 0.5045 &pm; 0.0217     | 0.7092 &pm; 0.0482     | 0.4867 &pm; 0.0308     | 0.6767 &pm; 0.0550     |
| PAiNN               | 2.1.0   |      250 | **0.4291 &pm; 0.0164** | **0.6014 &pm; 0.0238** | 0.4109 &pm; 0.0201     | 0.5782 &pm; 0.0317     |
| RGCN                | 2.2.0   |      800 | 0.5014 &pm; 0.0274     | 0.7028 &pm; 0.0332     | 0.4861 &pm; 0.0185     | 0.6776 &pm; 0.0261     |
| rGIN                | 3.0.0   |      300 | 0.6159 &pm; 0.0166     | 0.8445 &pm; 0.0147     | 0.5926 &pm; 0.0191     | 0.8060 &pm; 0.0172     |
| Schnet              | 2.2.2   |      800 | 0.4555 &pm; 0.0215     | 0.6473 &pm; 0.0541     | 0.4261 &pm; 0.0174     | 0.6026 &pm; 0.0473     |

#### LipopDataset

Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles. Graph labels for regression are octanol/water distribution coefficient (logD at pH 7.4). We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       | *Min. MAE*             | *Min. RMSE*            |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.4511 &pm; 0.0104     | 0.6193 &pm; 0.0149     | 0.4299 &pm; 0.0058     | 0.5958 &pm; 0.0112     |
| CMPNN       | 2.1.0   |      600 | 0.4129 &pm; 0.0069     | 0.5752 &pm; 0.0094     | 0.4016 &pm; 0.0091     | 0.5595 &pm; 0.0151     |
| DMPNN       | 2.1.0   |      300 | **0.3809 &pm; 0.0137** | **0.5503 &pm; 0.0251** | **0.3764 &pm; 0.0117** | **0.5452 &pm; 0.0237** |
| GAT         | 2.1.0   |      500 | 0.4954 &pm; 0.0172     | 0.6962 &pm; 0.0351     | 0.4756 &pm; 0.0144     | 0.6637 &pm; 0.0286     |
| GATv2       | 2.1.0   |      500 | 0.4081 &pm; 0.0099     | 0.5876 &pm; 0.0128     | 0.3910 &pm; 0.0097     | 0.5637 &pm; 0.0198     |
| GIN         | 2.1.0   |      300 | 0.4528 &pm; 0.0069     | 0.6382 &pm; 0.0286     | 0.4483 &pm; 0.0067     | 0.6308 &pm; 0.0277     |
| HamNet      | 2.1.0   |      400 | 0.4546 &pm; 0.0042     | 0.6293 &pm; 0.0139     | 0.4429 &pm; 0.0056     | 0.6123 &pm; 0.0118     |
| INorp       | 2.1.0   |      500 | 0.4635 &pm; 0.0106     | 0.6529 &pm; 0.0141     | 0.4549 &pm; 0.0113     | 0.6365 &pm; 0.0203     |
| MEGAN       | 2.1.0   |      400 | 0.3997 &pm; 0.0060     | 0.5635 &pm; 0.0114     | 0.3963 &pm; 0.0062     | 0.5589 &pm; 0.0126     |
| PAiNN       | 2.1.0   |      250 | 0.4033 &pm; 0.0123     | 0.5798 &pm; 0.0281     | 0.3997 &pm; 0.0112     | 0.5645 &pm; 0.0145     |
| Schnet      | 2.1.0   |      800 | 0.4788 &pm; 0.0046     | 0.6450 &pm; 0.0036     | 0.4739 &pm; 0.0041     | 0.6403 &pm; 0.0052     |

#### MatProjectJdft2dDataset

Materials Project dataset from Matbench with 636 crystal structures and their corresponding Exfoliation energy (meV/atom). We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [meV/atom]          | RMSE [meV/atom]           | *Min. MAE*              | *Min. RMSE*               |
|:-----------------------------|:--------|---------:|:------------------------|:--------------------------|:------------------------|:--------------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 42.6352 &pm; 9.6715     | **112.4714 &pm; 37.9213** | 40.2588 &pm; 9.9596     | 108.7458 &pm; 38.9984     |
| coGN                         | 3.0.1   |     1000 | **40.6968 &pm; 8.1630** | 114.9913 &pm; 34.0681     | **36.2421 &pm; 7.1143** | **101.3198 &pm; 36.1348** |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 49.2113 &pm; 12.7431    | 124.7198 &pm; 38.4492     | 46.9070 &pm; 13.2399    | 119.7063 &pm; 40.4388     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 56.5205 &pm; 10.8723    | 136.3116 &pm; 31.2617     | 46.1838 &pm; 11.0133    | 110.4555 &pm; 35.4563     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 50.5886 &pm; 9.9009     | 117.7118 &pm; 33.4786     | 46.6864 &pm; 11.7985    | 116.4087 &pm; 33.2183     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | 48.0629 &pm; 10.6137    | 121.6861 &pm; 36.7492     | 40.3236 &pm; 9.1278     | 112.7789 &pm; 37.5844     |

#### MatProjectPhononsDataset

Materials Project dataset from Matbench with 1,265 crystal structures and their corresponding vibration properties in [1/cm]. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV/atom]           | RMSE [eV/atom]          | *Min. MAE*              | *Min. RMSE*             |
|:-----------------------------|:--------|---------:|:------------------------|:------------------------|:------------------------|:------------------------|
| CGCNN.make_crystal_model     | 2.1.1   |     1000 | 46.1204 &pm; 3.2640     | 106.4514 &pm; 16.9401   | 45.4189 &pm; 3.1253     | 98.0387 &pm; 16.1605    |
| DimeNetPP.make_crystal_model | 2.1.1   |      780 | 36.7288 &pm; 1.3484     | 81.5038 &pm; 10.3550    | 35.0273 &pm; 0.8858     | 73.7587 &pm; 7.7229     |
| MEGAN                        | 2.1.1   |      400 | 50.3682 &pm; 7.2162     | 121.6629 &pm; 27.4599   | 46.7258 &pm; 4.3762     | 98.7136 &pm; 15.1310    |
| Megnet.make_crystal_model    | 2.1.0   |     1000 | **29.2085 &pm; 2.8130** | **53.9366 &pm; 7.0800** | **26.5068 &pm; 2.1145** | **48.0619 &pm; 7.1770** |
| NMPN.make_crystal_model      | 2.1.0   |      700 | 44.4253 &pm; 3.7905     | 91.1708 &pm; 23.8014    | 40.6537 &pm; 2.8499     | 81.0014 &pm; 21.7603    |
| PAiNN.make_crystal_model     | 2.1.1   |      800 | 47.2212 &pm; 3.8855     | 82.7834 &pm; 6.0730     | 45.7671 &pm; 3.7479     | 79.3576 &pm; 7.1055     |
| Schnet.make_crystal_model    | 2.2.3   |      800 | 40.2982 &pm; 1.6997     | 81.8959 &pm; 12.1697    | 39.8395 &pm; 1.7269     | 76.3132 &pm; 12.2788    |

#### MatProjectDielectricDataset

Materials Project dataset from Matbench with 4764 crystal structures and their corresponding Refractive index (unitless). We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [no unit]          | RMSE [no unit]         | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 0.3479 &pm; 0.0461     | 2.1384 &pm; 0.5135     | 0.3131 &pm; 0.0532     | 1.7784 &pm; 0.6500     |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 0.3337 &pm; 0.0608     | 1.8686 &pm; 0.6216     | 0.2923 &pm; 0.0542     | 1.7535 &pm; 0.6336     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 0.3485 &pm; 0.0443     | 2.0672 &pm; 0.5674     | 0.3086 &pm; 0.0506     | 1.7507 &pm; 0.6383     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.3587 &pm; 0.0518     | **1.8403 &pm; 0.6255** | 0.3059 &pm; 0.0483     | 1.7830 &pm; 0.6430     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | **0.3241 &pm; 0.0375** | 2.0324 &pm; 0.5585     | **0.2854 &pm; 0.0502** | **1.7323 &pm; 0.6378** |

#### MatProjectLogGVRHDataset

Materials Project dataset from Matbench with 10987 crystal structures and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average shear moduli in GPa. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [log(GPa)]         | RMSE [log(GPa)]        | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 0.0847 &pm; 0.0020     | 0.1286 &pm; 0.0044     | 0.0845 &pm; 0.0020     | 0.1281 &pm; 0.0043     |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 0.0805 &pm; 0.0027     | 0.1259 &pm; 0.0056     | **0.0785 &pm; 0.0027** | **0.1208 &pm; 0.0053** |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 0.0858 &pm; 0.0010     | 0.1337 &pm; 0.0035     | 0.0854 &pm; 0.0010     | 0.1326 &pm; 0.0036     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.0851 &pm; 0.0023     | 0.1284 &pm; 0.0057     | 0.0828 &pm; 0.0012     | 0.1233 &pm; 0.0038     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | **0.0798 &pm; 0.0011** | **0.1253 &pm; 0.0038** | 0.0793 &pm; 0.0011     | 0.1238 &pm; 0.0035     |

#### MatProjectLogKVRHDataset

Materials Project dataset from Matbench with 10987 crystal structures and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average bulk moduli in GPa. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [log(GPa)]         | RMSE [log(GPa)]        | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 0.0629 &pm; 0.0008     | 0.1198 &pm; 0.0037     | 0.0627 &pm; 0.0008     | 0.1182 &pm; 0.0036     |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | **0.0579 &pm; 0.0014** | **0.1120 &pm; 0.0045** | 0.0566 &pm; 0.0015     | 0.1078 &pm; 0.0057     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 0.0660 &pm; 0.0020     | 0.1251 &pm; 0.0058     | 0.0630 &pm; 0.0016     | 0.1193 &pm; 0.0044     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.0646 &pm; 0.0015     | 0.1177 &pm; 0.0052     | **0.0545 &pm; 0.0008** | **0.1055 &pm; 0.0055** |
| Schnet.make_crystal_model    | 2.2.2   |      800 | 0.0584 &pm; 0.0016     | 0.1143 &pm; 0.0062     | 0.0581 &pm; 0.0015     | 0.1118 &pm; 0.0055     |

#### MatProjectPerovskitesDataset

Materials Project dataset from Matbench with 18928 crystal structures and their corresponding Heat of formation of the entire 5-atom perovskite cell in eV. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV]               | RMSE [eV]              | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | **0.0316 &pm; 0.0012** | **0.0597 &pm; 0.0044** | **0.0316 &pm; 0.0012** | **0.0593 &pm; 0.0042** |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 0.0373 &pm; 0.0008     | 0.0660 &pm; 0.0038     | 0.0363 &pm; 0.0010     | 0.0642 &pm; 0.0035     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | 0.0351 &pm; 0.0013     | 0.0636 &pm; 0.0025     | 0.0349 &pm; 0.0013     | 0.0627 &pm; 0.0028     |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.0456 &pm; 0.0009     | 0.0742 &pm; 0.0024     | 0.0446 &pm; 0.0008     | 0.0729 &pm; 0.0021     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | 0.0347 &pm; 0.0007     | 0.0615 &pm; 0.0030     | 0.0347 &pm; 0.0007     | 0.0614 &pm; 0.0029     |

#### MatProjectGapDataset

Materials Project dataset from Matbench with 106113 crystal structures and their band gap as calculated by PBE DFT from the Materials Project, in eV. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV]               | RMSE [eV]              | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.2.2   |     1000 | 0.2298 &pm; 0.0054     | 0.5394 &pm; 0.0102     | 0.2293 &pm; 0.0055     | 0.5282 &pm; 0.0093     |
| DimeNetPP.make_crystal_model | 2.2.2   |      780 | 0.2089 &pm; 0.0022     | 0.4912 &pm; 0.0104     | 0.2061 &pm; 0.0021     | 0.4822 &pm; 0.0125     |
| Megnet.make_crystal_model    | 2.2.2   |     1000 | **0.2003 &pm; 0.0132** | **0.4839 &pm; 0.0303** | **0.1969 &pm; 0.0131** | **0.4707 &pm; 0.0284** |
| PAiNN.make_crystal_model     | 2.2.2   |      800 | 0.2220 &pm; 0.0037     | 0.5315 &pm; 0.0260     | 0.2215 &pm; 0.0035     | 0.5189 &pm; 0.0111     |
| Schnet.make_crystal_model    | 2.2.2   |      800 | 0.9351 &pm; 0.3720     | 1.5027 &pm; 0.4929     | 0.2940 &pm; 0.0290     | 0.6104 &pm; 0.0487     |

#### MatProjectIsMetalDataset

Materials Project dataset from Matbench with 106113 crystal structures and their corresponding Metallicity determined with pymatgen. 1 if the compound is a metal, 0 if the compound is not a metal. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | Accuracy               | AUC                    | *Max. Accuracy*        | *Max. AUC*       |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------|
| CGCNN.make_crystal_model     | 2.2.2   |      100 | 0.8902 &pm; 0.0021     | 0.9380 &pm; 0.0013     | 0.8930 &pm; 0.0012     | **nan &pm; nan** |
| DimeNetPP.make_crystal_model | 2.2.2   |       78 | **0.9067 &pm; 0.0021** | 0.9463 &pm; 0.0013     | **0.9104 &pm; 0.0023** | nan &pm; nan     |
| Megnet.make_crystal_model    | 2.2.2   |      100 | 0.9025 &pm; 0.0042     | **0.9559 &pm; 0.0027** | 0.9058 &pm; 0.0030     | nan &pm; nan     |
| PAiNN.make_crystal_model     | 2.2.2   |       80 | 0.8941 &pm; 0.0029     | 0.9331 &pm; 0.0024     | 0.8987 &pm; 0.0014     | nan &pm; nan     |
| Schnet.make_crystal_model    | 2.2.2   |       80 | 0.8937 &pm; 0.0045     | 0.9498 &pm; 0.0023     | 0.8997 &pm; 0.0030     | nan &pm; nan     |

#### MatProjectEFormDataset

Materials Project dataset from Matbench with 132752 crystal structures and their corresponding formation energy in [eV/atom]. We use a random 5-fold cross-validation. 

| model                        | kgcnn   |   epochs | MAE [eV/atom]          | RMSE [eV/atom]         | *Min. MAE*             | *Min. RMSE*            |
|:-----------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CGCNN.make_crystal_model     | 2.1.1   |     1000 | 0.0369 &pm; 0.0003     | 0.0873 &pm; 0.0026     | 0.0369 &pm; 0.0003     | 0.0870 &pm; 0.0025     |
| DimeNetPP.make_crystal_model | 2.1.1   |      780 | 0.0233 &pm; 0.0005     | 0.0644 &pm; 0.0020     | 0.0232 &pm; 0.0005     | 0.0627 &pm; 0.0015     |
| MEGAN                        | 2.1.1   |      800 | 0.0397 &pm; 0.0009     | 0.0902 &pm; 0.0041     | 0.0388 &pm; 0.0012     | 0.0893 &pm; 0.0043     |
| Megnet.make_crystal_model    | 2.1.0   |     1000 | 0.0247 &pm; 0.0006     | 0.0639 &pm; 0.0028     | 0.0247 &pm; 0.0006     | 0.0628 &pm; 0.0026     |
| PAiNN.make_crystal_model     | 2.1.1   |      800 | 0.0244 &pm; 0.0002     | 0.0568 &pm; 0.0032     | 0.0244 &pm; 0.0002     | 0.0562 &pm; 0.0029     |
| Schnet.make_crystal_model    | 2.1.1   |      800 | **0.0215 &pm; 0.0003** | **0.0525 &pm; 0.0030** | **0.0215 &pm; 0.0003** | **0.0516 &pm; 0.0025** |

#### MutagenicityDataset

Mutagenicity dataset from TUDataset for classification with 4337 graphs. The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.7397 &pm; 0.0111     | 0.8207 &pm; 0.0111     | 0.8185 &pm; 0.0114     | 0.8836 &pm; 0.0054     |
| CMPNN       | 2.1.0   |      600 | 0.8102 &pm; 0.0157     | 0.8348 &pm; 0.0237     | 0.8218 &pm; 0.0138     | 0.8832 &pm; 0.0090     |
| DMPNN       | 2.1.0   |      300 | **0.8296 &pm; 0.0126** | 0.8714 &pm; 0.0075     | **0.8439 &pm; 0.0075** | **0.9015 &pm; 0.0066** |
| GAT         | 2.1.0   |      500 | 0.8008 &pm; 0.0115     | 0.8294 &pm; 0.0113     | 0.8165 &pm; 0.0067     | 0.8817 &pm; 0.0068     |
| GATv2       | 2.1.0   |      500 | 0.8029 &pm; 0.0122     | 0.8337 &pm; 0.0046     | 0.8278 &pm; 0.0043     | 0.8791 &pm; 0.0053     |
| GIN         | 2.1.0   |      300 | 0.8185 &pm; 0.0127     | **0.8734 &pm; 0.0094** | 0.8349 &pm; 0.0090     | 0.8942 &pm; 0.0050     |
| GraphSAGE   | 2.1.0   |      500 | 0.8165 &pm; 0.0061     | 0.8530 &pm; 0.0089     | 0.8354 &pm; 0.0058     | 0.8941 &pm; 0.0039     |
| INorp       | 2.1.0   |      500 | 0.7955 &pm; 0.0037     | 0.8255 &pm; 0.0047     | 0.8139 &pm; 0.0075     | 0.8637 &pm; 0.0118     |
| MEGAN       | 2.1.1   |      500 | 0.8137 &pm; 0.0117     | 0.8591 &pm; 0.0077     | 0.8303 &pm; 0.0074     | 0.8915 &pm; 0.0051     |

#### MUTAGDataset

MUTAG dataset from TUDataset for classification with 188 graphs. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.8085 &pm; 0.1031     | 0.8471 &pm; 0.0890     | 0.9100 &pm; 0.0753     | 0.9442 &pm; 0.0479     |
| CMPNN       | 2.1.0   |      600 | 0.7873 &pm; 0.0724     | 0.7811 &pm; 0.0762     | 0.8778 &pm; 0.0754     | 0.9137 &pm; 0.0648     |
| DMPNN       | 2.1.0   |      300 | 0.8461 &pm; 0.0474     | 0.8686 &pm; 0.0480     | **0.9203 &pm; 0.0439** | **0.9594 &pm; 0.0364** |
| GAT         | 2.1.0   |      500 | 0.8351 &pm; 0.0920     | 0.8779 &pm; 0.0854     | 0.8671 &pm; 0.0745     | 0.9370 &pm; 0.0613     |
| GATv2       | 2.1.0   |      500 | 0.8144 &pm; 0.0757     | 0.8400 &pm; 0.0688     | 0.8622 &pm; 0.0443     | 0.9352 &pm; 0.0364     |
| GIN         | 2.1.0   |      300 | **0.8512 &pm; 0.0485** | **0.8861 &pm; 0.0922** | 0.9098 &pm; 0.0696     | 0.9537 &pm; 0.0574     |
| GraphSAGE   | 2.1.0   |      500 | 0.8193 &pm; 0.0445     | 0.8560 &pm; 0.0651     | 0.8939 &pm; 0.0723     | 0.9137 &pm; 0.0618     |
| INorp       | 2.1.0   |      500 | 0.8407 &pm; 0.0829     | 0.8549 &pm; 0.0705     | 0.8936 &pm; 0.0577     | 0.9190 &pm; 0.0482     |
| MEGAN       | 2.1.1   |      500 | 0.7977 &pm; 0.0663     | 0.8810 &pm; 0.0568     | 0.8778 &pm; 0.0611     | 0.9328 &pm; 0.0532     |

#### FreeSolvDataset

FreeSolv (MoleculeNet) consists of 642 compounds as smiles and their corresponding hydration free energy for small neutral molecules in water. We use a random 5-fold cross-validation. 

| model               | kgcnn   |   epochs | MAE [log mol/L]        | RMSE [log mol/L]       | *Min. MAE*             | *Min. RMSE*            |
|:--------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP         | 2.1.0   |      200 | 0.5853 &pm; 0.0519     | 1.0168 &pm; 0.1386     | 0.5283 &pm; 0.0526     | 0.9140 &pm; 0.1025     |
| CMPNN               | 2.1.0   |      600 | 0.5319 &pm; 0.0655     | 0.9262 &pm; 0.1597     | 0.4983 &pm; 0.0589     | 0.8673 &pm; 0.1571     |
| DimeNetPP           | 2.1.0   |      300 | 0.5791 &pm; 0.0649     | 0.9439 &pm; 0.1602     | 0.5519 &pm; 0.0648     | 0.8618 &pm; 0.0973     |
| DMPNN               | 2.1.0   |      300 | 0.5305 &pm; 0.0474     | **0.9070 &pm; 0.1497** | **0.4809 &pm; 0.0450** | **0.8192 &pm; 0.1484** |
| GAT                 | 2.1.0   |      500 | 0.5970 &pm; 0.0776     | 1.0107 &pm; 0.1554     | 0.5750 &pm; 0.0684     | 0.9780 &pm; 0.1493     |
| GATv2               | 2.1.0   |      500 | 0.6390 &pm; 0.0467     | 1.1203 &pm; 0.1491     | 0.5988 &pm; 0.0355     | 0.9891 &pm; 0.1020     |
| GCN                 | 2.1.0   |      800 | 0.7766 &pm; 0.0774     | 1.3245 &pm; 0.2008     | 0.7176 &pm; 0.0542     | 1.1710 &pm; 0.0881     |
| GIN                 | 2.1.0   |      300 | 0.7161 &pm; 0.0492     | 1.1171 &pm; 0.1233     | 0.6293 &pm; 0.0487     | 1.0069 &pm; 0.1053     |
| GIN.make_model_edge | 2.1.0   |      300 | 0.6285 &pm; 0.0588     | 1.0457 &pm; 0.1458     | 0.5819 &pm; 0.0585     | 0.9515 &pm; 0.1362     |
| GraphSAGE           | 2.1.0   |      500 | 0.5667 &pm; 0.0577     | 0.9861 &pm; 0.1328     | 0.5496 &pm; 0.0575     | 0.9236 &pm; 0.1444     |
| HamNet              | 2.1.0   |      400 | 0.6395 &pm; 0.0496     | 1.0508 &pm; 0.0827     | 0.5862 &pm; 0.0446     | 0.9691 &pm; 0.0884     |
| INorp               | 2.1.0   |      500 | 0.6448 &pm; 0.0607     | 1.0911 &pm; 0.1530     | 0.6021 &pm; 0.0640     | 0.9915 &pm; 0.1678     |
| MAT                 | 2.1.1   |      400 | 0.8477 &pm; 0.0488     | 1.2582 &pm; 0.0810     | 0.8325 &pm; 0.0405     | 1.2045 &pm; 0.0701     |
| MEGAN               | 2.1.1   |      400 | 0.5689 &pm; 0.0735     | 0.9689 &pm; 0.1602     | 0.5639 &pm; 0.0730     | 0.9435 &pm; 0.1704     |
| Megnet              | 2.1.0   |      800 | 0.9749 &pm; 0.0429     | 1.5328 &pm; 0.0862     | 0.8850 &pm; 0.0481     | 1.3589 &pm; 0.0661     |
| NMPN                | 2.1.0   |      800 | 0.5733 &pm; 0.0392     | 0.9861 &pm; 0.0816     | 0.5416 &pm; 0.0375     | 0.9080 &pm; 0.0851     |
| PAiNN               | 2.1.0   |      250 | **0.5128 &pm; 0.0565** | 0.9403 &pm; 0.1387     | 0.4854 &pm; 0.0474     | 0.8569 &pm; 0.1270     |
| Schnet              | 2.1.0   |      800 | 0.5980 &pm; 0.0556     | 1.0614 &pm; 0.1531     | 0.5616 &pm; 0.0456     | 0.9441 &pm; 0.1021     |

#### PROTEINSDataset

TUDataset of proteins that are classified as enzymes or non-enzymes. Nodes represent the amino acids of the protein. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.0   |      200 | 0.7296 &pm; 0.0126     | 0.7967 &pm; 0.0264     | 0.7853 &pm; 0.0185     | **0.8388 &pm; 0.0345** |
| CMPNN       | 2.1.0   |      600 | 0.7377 &pm; 0.0164     | 0.7532 &pm; 0.0174     | 0.7763 &pm; 0.0210     | 0.8337 &pm; 0.0252     |
| DMPNN       | 2.1.0   |      300 | 0.7395 &pm; 0.0300     | **0.8038 &pm; 0.0365** | **0.7907 &pm; 0.0272** | 0.8357 &pm; 0.0274     |
| GAT         | 2.1.0   |      500 | 0.7314 &pm; 0.0283     | 0.7884 &pm; 0.0404     | 0.7790 &pm; 0.0366     | 0.8281 &pm; 0.0366     |
| GATv2       | 2.1.0   |      500 | 0.6999 &pm; 0.0266     | 0.7137 &pm; 0.0177     | 0.7835 &pm; 0.0348     | 0.8270 &pm; 0.0387     |
| GIN         | 2.1.0   |      150 | 0.7098 &pm; 0.0357     | 0.7437 &pm; 0.0454     | 0.7449 &pm; 0.0325     | 0.7728 &pm; 0.0401     |
| GraphSAGE   | 2.1.0   |      500 | 0.6937 &pm; 0.0273     | 0.7263 &pm; 0.0391     | 0.7718 &pm; 0.0374     | 0.8011 &pm; 0.0325     |
| INorp       | 2.1.0   |      500 | 0.7242 &pm; 0.0359     | 0.7333 &pm; 0.0228     | 0.7763 &pm; 0.0295     | 0.8178 &pm; 0.0327     |
| MEGAN       | 2.1.1   |      200 | **0.7449 &pm; 0.0222** | 0.8015 &pm; 0.0195     | 0.7817 &pm; 0.0289     | 0.8336 &pm; 0.0306     |

#### Tox21MolNetDataset

Tox21 (MoleculeNet) consists of 7831 compounds as smiles and 12 different targets relevant to drug toxicity. We use random 5-fold cross-validation. 

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | BACC                   | *Max. BACC*            | *Max. Accuracy*        | *Max. AUC*             |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 2.2.1   |       50 | 0.9352 &pm; 0.0022     | 0.8127 &pm; 0.0100     | 0.6872 &pm; 0.0096     | 0.7176 &pm; 0.0070     | 0.9404 &pm; 0.0017     | **0.8446 &pm; 0.0058** |
| CMPNN       | 2.2.1   |       30 | 0.9311 &pm; 0.0050     | 0.7769 &pm; 0.0344     | 0.6127 &pm; 0.0724     | 0.6213 &pm; 0.0764     | 0.9324 &pm; 0.0057     | 0.7901 &pm; 0.0458     |
| DMPNN       | 2.2.1   |       50 | **0.9385 &pm; 0.0015** | 0.8295 &pm; 0.0103     | 0.6906 &pm; 0.0069     | 0.7133 &pm; 0.0082     | **0.9407 &pm; 0.0015** | 0.8412 &pm; 0.0105     |
| GAT         | 2.2.1   |       50 | 0.9365 &pm; 0.0019     | 0.8309 &pm; 0.0053     | 0.6540 &pm; 0.0102     | 0.6742 &pm; 0.0085     | 0.9375 &pm; 0.0021     | 0.8373 &pm; 0.0079     |
| GATv2       | 2.2.1   |       50 | 0.9366 &pm; 0.0019     | 0.8305 &pm; 0.0051     | 0.6775 &pm; 0.0093     | 0.6955 &pm; 0.0110     | 0.9381 &pm; 0.0017     | 0.8390 &pm; 0.0035     |
| GIN         | 2.2.1   |       50 | 0.9358 &pm; 0.0031     | 0.8284 &pm; 0.0095     | 0.6986 &pm; 0.0129     | 0.7135 &pm; 0.0048     | 0.9392 &pm; 0.0022     | 0.8391 &pm; 0.0097     |
| GraphSAGE   | 2.2.1   |      100 | 0.9286 &pm; 0.0042     | 0.8092 &pm; 0.0079     | **0.7056 &pm; 0.0087** | **0.7226 &pm; 0.0032** | 0.9385 &pm; 0.0025     | 0.8385 &pm; 0.0086     |
| INorp       | 2.2.1   |       50 | 0.9335 &pm; 0.0032     | 0.8256 &pm; 0.0080     | 0.6854 &pm; 0.0119     | 0.7004 &pm; 0.0030     | 0.9372 &pm; 0.0015     | 0.8327 &pm; 0.0081     |
| MEGAN       | 2.2.1   |       50 | 0.9374 &pm; 0.0033     | **0.8389 &pm; 0.0094** | 0.6627 &pm; 0.0133     | 0.6849 &pm; 0.0041     | 0.9383 &pm; 0.0027     | 0.8436 &pm; 0.0102     |
| Schnet      | 2.2.1   |       50 | 0.9336 &pm; 0.0026     | 0.7856 &pm; 0.0054     | 0.6591 &pm; 0.0144     | 0.6832 &pm; 0.0061     | 0.9356 &pm; 0.0026     | 0.8054 &pm; 0.0055     |

#### ClinToxDataset

ClinTox (MoleculeNet) consists of 1478 compounds as smiles and data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons. We use random 5-fold cross-validation. The first label 'approved' is chosen as target.

| model       | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| AttentiveFP | 2.1.1   |       50 | 0.9372 &pm; 0.0095     | 0.8317 &pm; 0.0426     | 0.9568 &pm; 0.0058     | 0.9073 &pm; 0.0352     |
| CMPNN       | 2.1.1   |       30 | 0.9365 &pm; 0.0216     | 0.8067 &pm; 0.0670     | 0.9574 &pm; 0.0122     | 0.8880 &pm; 0.0442     |
| DMPNN       | 2.1.1   |       50 | 0.9385 &pm; 0.0146     | **0.8519 &pm; 0.0271** | **0.9588 &pm; 0.0066** | **0.9234 &pm; 0.0313** |
| GAT         | 2.1.1   |       50 | 0.9338 &pm; 0.0164     | 0.8354 &pm; 0.0487     | 0.9568 &pm; 0.0078     | 0.8835 &pm; 0.0344     |
| GATv2       | 2.1.1   |       50 | 0.9378 &pm; 0.0087     | 0.8331 &pm; 0.0663     | 0.9554 &pm; 0.0099     | 0.8938 &pm; 0.0274     |
| GIN         | 2.1.1   |       50 | 0.9277 &pm; 0.0139     | 0.8244 &pm; 0.0478     | 0.9554 &pm; 0.0066     | 0.8966 &pm; 0.0362     |
| GraphSAGE   | 2.1.1   |      100 | 0.9385 &pm; 0.0099     | 0.7795 &pm; 0.0744     | 0.9554 &pm; 0.0078     | 0.9026 &pm; 0.0275     |
| INorp       | 2.1.1   |       50 | 0.9304 &pm; 0.0106     | 0.7826 &pm; 0.0573     | 0.9547 &pm; 0.0087     | 0.8793 &pm; 0.0456     |
| MEGAN       | 2.1.1   |       50 | **0.9493 &pm; 0.0130** | 0.8394 &pm; 0.0608     | 0.9568 &pm; 0.0087     | 0.8952 &pm; 0.0288     |
| Schnet      | 2.1.1   |       50 | 0.9318 &pm; 0.0078     | 0.6807 &pm; 0.0745     | 0.9392 &pm; 0.0074     | 0.7770 &pm; 0.0519     |

#### QM7Dataset

QM7 dataset is a subset of GDB-13. Molecules of up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. We use dataset-specific 5-fold cross-validation. The atomization energies are given in kcal/mol and are ranging from -800 to -2000 kcal/mol). 

| model     | kgcnn   |   epochs | MAE [kcal/mol]         | RMSE [kcal/mol]        | *Min. MAE*             | *Min. RMSE*            |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DimeNetPP | 2.1.1   |      872 | 2.7266 &pm; 0.1022     | 6.1305 &pm; 0.9606     | 2.7223 &pm; 0.1024     | 6.0729 &pm; 0.9542     |
| EGNN      | 2.1.1   |      800 | 1.6182 &pm; 0.1712     | 3.8677 &pm; 0.7640     | 1.5839 &pm; 0.1631     | 3.8364 &pm; 0.7505     |
| HDNNP2nd  | 2.2.0   |      500 | 12.3555 &pm; 2.6972    | 25.6856 &pm; 11.3776   | 12.3555 &pm; 2.6972    | 21.7600 &pm; 7.7237    |
| MEGAN     | 2.1.1   |      800 | 10.4494 &pm; 1.6076    | 11.5596 &pm; 1.5710    | 3.8778 &pm; 0.3620     | 6.1670 &pm; 0.6990     |
| Megnet    | 2.1.1   |      800 | 1.4626 &pm; 0.0818     | 3.1522 &pm; 0.2409     | 1.4497 &pm; 0.0796     | 3.0018 &pm; 0.1757     |
| MXMNet    | 2.1.1   |      900 | **1.1078 &pm; 0.0799** | **2.8693 &pm; 0.7399** | **1.0586 &pm; 0.0968** | **2.6226 &pm; 0.7498** |
| NMPN      | 2.1.1   |      500 | 6.4698 &pm; 0.8256     | 35.0397 &pm; 4.3985    | 6.4243 &pm; 0.8372     | 31.7830 &pm; 4.8717    |
| PAiNN     | 2.1.1   |      872 | 1.2715 &pm; 0.0235     | 4.4958 &pm; 1.8048     | 1.2518 &pm; 0.0198     | 4.4525 &pm; 1.7880     |
| Schnet    | 2.1.1   |      800 | 2.5840 &pm; 0.3479     | 10.3788 &pm; 9.1047    | 2.5840 &pm; 0.3479     | 8.4264 &pm; 6.3761     |

#### QM9Dataset

QM9 dataset of 134k stable small organic molecules made up of C,H,O,N,F. Labels include geometric, energetic, electronic, and thermodynamic properties. We use a random 10-fold cross-validation, but not all splits are evaluated for cheaper evaluation. Test errors are MAE and for energies are given in [eV]. 

| model     | kgcnn   |   epochs | HOMO [eV]              | LUMO [eV]              | U0 [eV]                | H [eV]                 | G [eV]                 |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DimeNetPP | 2.1.0   |      600 | 0.0242 &pm; 0.0006     | 0.0209 &pm; 0.0002     | 0.0073 &pm; 0.0003     | **0.0073 &pm; 0.0003** | 0.0084 &pm; 0.0004     |
| EGNN      | 2.1.1   |      800 | 0.0273 &pm; 0.0004     | 0.0226 &pm; 0.0011     | 0.0081 &pm; 0.0002     | 0.0090 &pm; 0.0004     | 0.0095 &pm; 0.0005     |
| Megnet    | 2.1.0   |      800 | 0.0423 &pm; 0.0014     | 0.0354 &pm; 0.0008     | 0.0136 &pm; 0.0006     | 0.0135 &pm; 0.0001     | 0.0140 &pm; 0.0002     |
| MXMNet    | 2.1.1   |      900 | **0.0238 &pm; 0.0012** | **0.0203 &pm; 0.0007** | **0.0067 &pm; 0.0001** | 0.0074 &pm; 0.0008     | **0.0079 &pm; 0.0008** |
| NMPN      | 2.1.0   |      700 | 0.0627 &pm; 0.0013     | 0.0618 &pm; 0.0006     | 0.0385 &pm; 0.0011     | 0.0382 &pm; 0.0005     | 0.0365 &pm; 0.0005     |
| PAiNN     | 2.1.0   |      872 | 0.0287 &pm; 0.0068     | 0.0230 &pm; 0.0005     | 0.0075 &pm; 0.0002     | 0.0075 &pm; 0.0003     | 0.0087 &pm; 0.0002     |
| Schnet    | 2.1.0   |      800 | 0.0351 &pm; 0.0005     | 0.0293 &pm; 0.0006     | 0.0116 &pm; 0.0004     | 0.0117 &pm; 0.0004     | 0.0120 &pm; 0.0002     |

#### SIDERDataset

SIDER (MoleculeNet) consists of 1427 compounds as smiles and data for adverse drug reactions (ADR), grouped into 27 system organ classes. We use random 5-fold cross-validation.

| model     | kgcnn   |   epochs | Accuracy               | AUC(ROC)               | *Max. Accuracy*        | *Max. AUC*             |
|:----------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| CMPNN     | 2.1.0   |       30 | 0.7360 &pm; 0.0048     | 0.5729 &pm; 0.0303     | 0.7567 &pm; 0.0045     | 0.5874 &pm; 0.0278     |
| DMPNN     | 2.1.0   |       50 | 0.6866 &pm; 0.1280     | 0.5942 &pm; 0.0508     | 0.6977 &pm; 0.1302     | 0.6028 &pm; 0.0489     |
| GAT       | 2.1.0   |       50 | 0.7559 &pm; 0.0078     | 0.6064 &pm; 0.0209     | 0.7616 &pm; 0.0038     | 0.6151 &pm; 0.0170     |
| GATv2     | 2.1.0   |       50 | 0.7515 &pm; 0.0066     | 0.6026 &pm; 0.0199     | 0.7597 &pm; 0.0031     | 0.6087 &pm; 0.0166     |
| GIN       | 2.1.0   |       50 | 0.7438 &pm; 0.0075     | 0.6109 &pm; 0.0256     | 0.7627 &pm; 0.0045     | **0.6365 &pm; 0.0227** |
| GraphSAGE | 2.1.0   |       30 | 0.7542 &pm; 0.0080     | 0.5946 &pm; 0.0151     | 0.7606 &pm; 0.0025     | 0.6031 &pm; 0.0166     |
| INorp     | 2.1.0   |       50 | 0.7471 &pm; 0.0105     | 0.5836 &pm; 0.0211     | 0.7594 &pm; 0.0027     | 0.6045 &pm; 0.0084     |
| MEGAN     | 2.1.1   |      150 | 0.7440 &pm; 0.0077     | **0.6186 &pm; 0.0160** | **0.7638 &pm; 0.0031** | 0.6307 &pm; 0.0156     |
| Schnet    | 2.1.0   |       50 | **0.7581 &pm; 0.0037** | 0.6075 &pm; 0.0143     | 0.7614 &pm; 0.0051     | 0.6127 &pm; 0.0152     |

#### MD17Dataset

Energies and forces for molecular dynamics trajectories of eight organic molecules. All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. Errors are MAE for forces. Results are for the CCSD and CCSD(T) data in MD17. 

| model                      | kgcnn   |   epochs | Aspirin             | Toluene             | Malonaldehyde       | Benzene             | Ethanol             |
|:---------------------------|:--------|---------:|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|
| DimeNetPP.EnergyForceModel | 2.2.0   |     1000 | **0.5366 &pm; nan** | **0.2380 &pm; nan** | **0.3653 &pm; nan** | 0.0861 &pm; nan     | **0.2221 &pm; nan** |
| EGNN.EnergyForceModel      | 2.2.2   |     1000 | 1.8978 &pm; nan     | 0.9314 &pm; nan     | 0.9255 &pm; nan     | 0.3273 &pm; nan     | 0.5286 &pm; nan     |
| Megnet.EnergyForceModel    | 2.2.0   |     1000 | 2.2431 &pm; nan     | 1.0476 &pm; nan     | 1.7242 &pm; nan     | 0.5225 &pm; nan     | 1.4967 &pm; nan     |
| MXMNet.EnergyForceModel    | 2.2.0   |     1000 | 1.3700 &pm; nan     | 0.5998 &pm; nan     | 0.7752 &pm; nan     | 0.3669 &pm; nan     | 0.4451 &pm; nan     |
| NMPN.EnergyForceModel      | 2.2.0   |     1000 | 1.1429 &pm; nan     | 0.6937 &pm; nan     | 0.6134 &pm; nan     | 0.4112 &pm; nan     | 0.3220 &pm; nan     |
| PAiNN.EnergyForceModel     | 2.2.2   |     1000 | 0.8388 &pm; nan     | 0.2704 &pm; nan     | 0.7121 &pm; nan     | **0.0448 &pm; nan** | 0.5373 &pm; nan     |
| Schnet.EnergyForceModel    | 2.2.2   |     1000 | 1.0816 &pm; nan     | 0.6011 &pm; nan     | 0.5777 &pm; nan     | 0.2924 &pm; nan     | 0.4020 &pm; nan     |

#### MD17RevisedDataset

Energies and forces for molecular dynamics trajectories. All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. Errors are MAE for forces. 

| model                      | kgcnn   |   epochs | Aspirin                | Toluene                | Malonaldehyde          | Benzene                | Ethanol                |
|:---------------------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| DimeNetPP.EnergyForceModel | 2.2.0   |     1000 | **0.5605 &pm; 0.0201** | **0.2207 &pm; 0.0117** | **0.4053 &pm; 0.0107** | 0.0656 &pm; 0.0055     | **0.2447 &pm; 0.0135** |
| EGNN.EnergyForceModel      | 2.2.2   |     1000 | 2.0576 &pm; 0.1748     | 0.8262 &pm; 0.0383     | 1.0048 &pm; 0.0401     | 0.3059 &pm; 0.0141     | 0.5360 &pm; 0.0365     |
| Megnet.EnergyForceModel    | 2.2.0   |     1000 | 2.3214 &pm; 0.2942     | 3.8695 &pm; 5.2614     | 1.6904 &pm; 0.1626     | 0.5341 &pm; 0.0907     | 1.2936 &pm; 0.0536     |
| MXMNet.EnergyForceModel    | 2.2.0   |     1000 | 1.8941 &pm; 0.0502     | 1.0880 &pm; 0.0628     | 1.2041 &pm; 0.0399     | 0.3573 &pm; 0.0302     | 0.6136 &pm; 0.0297     |
| NMPN.EnergyForceModel      | 2.2.0   |     1000 | 1.0653 &pm; 0.0263     | 0.6971 &pm; 0.0772     | 0.6197 &pm; 0.0327     | 0.3596 &pm; 0.0401     | 0.3444 &pm; 0.0219     |
| PAiNN.EnergyForceModel     | 2.2.2   |     1000 | 0.7901 &pm; 0.0062     | 0.2497 &pm; 0.0049     | 0.7496 &pm; 0.0109     | **0.0414 &pm; 0.0014** | 0.5676 &pm; 0.0215     |
| Schnet.EnergyForceModel    | 2.2.2   |     1000 | 0.9862 &pm; 0.0095     | 0.5378 &pm; 0.0036     | 0.6461 &pm; 0.0093     | 0.2521 &pm; 0.0074     | 0.4270 &pm; 0.0115     |

#### ISO17Dataset

The database consist of 129 molecules each containing 5,000 conformational geometries, energies and forces with a resolution of 1 femtosecond in the molecular dynamics trajectories. The molecules were randomly drawn from the largest set of isomers in the QM9 dataset. 

| model                   | kgcnn   |   epochs | Energy (test_within)   | Force (test_within)   | *Min. Energy* (test_within)   | *Min. Force* (test_within)   |
|:------------------------|:--------|---------:|:-----------------------|:----------------------|:------------------------------|:-----------------------------|
| Schnet.EnergyForceModel | 2.2.2   |     1000 | **0.0059 &pm; nan**    | **0.0132 &pm; nan**   | **0.0058 &pm; nan**           | **0.0132 &pm; nan**          |

#### VgdMockDataset

Synthetic classification dataset containing 100 small, randomly generated graphs, where half of them were seeded with a triangular subgraph motif, which is the explanation ground truth for the target class distinction.

| model            | kgcnn   |   epochs | Categorical Accuracy   | Node AUC               | Edge AUC               |
|:-----------------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|
| GCN_GnnExplainer | 2.2.1   |      100 | 0.8700 &pm; 0.1122     | 0.7621 &pm; 0.0357     | 0.6051 &pm; 0.0416     |
| MEGAN            | 2.2.0   |      100 | **0.9400 &pm; 0.0490** | **0.8873 &pm; 0.0250** | **0.9518 &pm; 0.0241** |

#### VgdRbMotifsDataset

Synthetic graph regression dataset consisting of 5000 small, randomly generated graphs, where some of them are seeded with special red- or blue-dominated subgraph motifs, where blue motifs contribute negatively to a graph's overall target value and red motifs contribute positively. The explanation ground truth for this datasets consists of these motifs.

| model   | kgcnn   |   epochs | MSE                    | Node AUC               | Edge AUC               |
|:--------|:--------|---------:|:-----------------------|:-----------------------|:-----------------------|
| MEGAN   | 2.2.0   |      100 | **0.2075 &pm; 0.0421** | **0.9051 &pm; 0.0130** | **0.8096 &pm; 0.0414** |

