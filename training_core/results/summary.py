import os
import argparse
import numpy as np
import pandas as pd
import yaml
from math import nan

parser = argparse.ArgumentParser(description='Summary of training stats.')
parser.add_argument("--min_max", required=False, help="Show min/max values for stats.", default=False, type=bool)
args = vars(parser.parse_args())
print("Input of argparse:", args)
show_min_max = args["min_max"]

benchmark_datasets = {
    # "CoraLuDataset": {
    #     "general_info": [
    #         "Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse attributes and 7 node classes. ",
    #         "Here we use random 5-fold cross-validation on nodes. ",
    #     ],
    #     "targets": [
    #         {"metric": "val_categorical_accuracy", "name": "Categorical accuracy", "find_best": "max"},
    #         {"metric": "max_val_categorical_accuracy", "name": "*Max. Categorical accuracy*", "find_best": "max",
    #          "is_min_max": True},
    #     ]
    # },
    # "CoraDataset": {
    #     "general_info": [
    #         "Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. ",
    #         "Here we use random 5-fold cross-validation on nodes. ",
    #     ],
    #     "targets": [
    #         {"metric": "val_categorical_accuracy", "name": "Categorical accuracy", "find_best": "max"},
    #         {"metric": "max_val_categorical_accuracy", "name": "*Max. Categorical accuracy*", "find_best": "max",
    #          "is_min_max": True},
    #     ]
    # },
    "ESOLDataset": {
        "general_info": [
            "ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). ",
            "We use random 5-fold cross-validation. "
        ],
        "targets": [
            {"metric": "val_scaled_mean_absolute_error", "name": "MAE [log mol/L]", "find_best": "min"},
            {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log mol/L]", "find_best": "min"},
            {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
             "is_min_max": True},
            {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
             "is_min_max": True}
        ]
    },
    # "LipopDataset": {
    #     "general_info": [
    #         "Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles. ",
    #         "Graph labels for regression are octanol/water distribution coefficient (logD at pH 7.4). ",
    #         "We use random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [log mol/L]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log mol/L]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectJdft2dDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 636 crystal structures ",
    #         "and their corresponding Exfoliation energy (meV/atom). ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [meV/atom]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [meV/atom]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectPhononsDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 1,265 crystal structures ",
    #         "and their corresponding vibration properties in [1/cm]. ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [eV/atom]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [eV/atom]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectDielectricDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 4764 crystal structures ",
    #         "and their corresponding Refractive index (unitless). ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [no unit]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [no unit]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectLogGVRHDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 10987 crystal structures ",
    #         "and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average shear moduli in GPa. ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [log(GPa)]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log(GPa)]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectLogKVRHDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 10987 crystal structures ",
    #         "and their corresponding Base 10 logarithm of the DFT Voigt-Reuss-Hill average bulk moduli in GPa. ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [log(GPa)]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log(GPa)]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectPerovskitesDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 18928 crystal structures ",
    #         "and their corresponding Heat of formation of the entire 5-atom perovskite cell in eV. ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [eV]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [eV]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectGapDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 106113 crystal structures ",
    #         "and their band gap as calculated by PBE DFT from the Materials Project, in eV. ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [eV]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [eV]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectIsMetalDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 106113 crystal structures ",
    #         "and their corresponding Metallicity determined with pymatgen. ",
    #         "1 if the compound is a metal, 0 if the compound is not a metal. ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_acc", "name": "Accuracy", "find_best": "max"},
    #         {"metric": "val_auc", "name": "AUC", "find_best": "max"},
    #         {"metric": "max_val_acc", "name": "*Max. Accuracy*", "find_best": "max",
    #          "is_min_max": True},
    #         {"metric": "max_val_uac", "name": "*Max. AUC*", "find_best": "max",
    #          "is_min_max": True}
    #     ]
    # },
    # "MatProjectEFormDataset": {
    #     "general_info": [
    #         "Materials Project dataset from Matbench with 132752 crystal structures ",
    #         "and their corresponding formation energy in [eV/atom]. ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [eV/atom]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [eV/atom]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "MutagenicityDataset": {
    #     "general_info": [
    #         "Mutagenicity dataset from TUDataset for classification with 4337 graphs. ",
    #         "The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_accuracy", "name": "Accuracy", "find_best": "max"},
    #         {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"},
    #         {"metric": "max_val_accuracy", "name": "*Max. Accuracy*", "find_best": "max", "is_min_max": True},
    #         {"metric": "max_val_auc", "name": "*Max. AUC*", "find_best": "max", "is_min_max": True}
    #     ]
    # },
    # "MUTAGDataset": {
    #     "general_info": [
    #         "MUTAG dataset from TUDataset for classification with 188 graphs. ",
    #         "We use random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_accuracy", "name": "Accuracy", "find_best": "max"},
    #         {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"},
    #         {"metric": "max_val_accuracy", "name": "*Max. Accuracy*", "find_best": "max", "is_min_max": True},
    #         {"metric": "max_val_auc", "name": "*Max. AUC*", "find_best": "max", "is_min_max": True}
    #     ]
    # },
    # "FreeSolvDataset": {
    #     "general_info": [
    #         "FreeSolv (MoleculeNet) consists of 642 compounds as smiles and ",
    #         "their corresponding hydration free energy for small neutral molecules in water. ",
    #         "We use a random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [log mol/L]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log mol/L]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "PROTEINSDataset": {
    #     "general_info": [
    #         "TUDataset of proteins that are classified as enzymes or non-enzymes. ",
    #         "Nodes represent the amino acids of the protein. ",
    #         "We use random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_accuracy", "name": "Accuracy", "find_best": "max"},
    #         {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"},
    #         {"metric": "max_val_accuracy", "name": "*Max. Accuracy*", "find_best": "max", "is_min_max": True},
    #         {"metric": "max_val_auc", "name": "*Max. AUC*", "find_best": "max", "is_min_max": True}
    #     ]
    # },
    # "Tox21MolNetDataset": {
    #     "general_info": [
    #         "Tox21 (MoleculeNet) consists of 7831 compounds as smiles and ",
    #         "12 different targets relevant to drug toxicity. ",
    #         "We use random 5-fold cross-validation. "
    #     ],
    #     "targets": [
    #         {"metric": "val_binary_accuracy_no_nan", "name": "Accuracy", "find_best": "max"},
    #         {"metric": "val_AUC_no_nan", "name": "AUC(ROC)", "find_best": "max"},
    #         {"metric": "val_balanced_binary_accuracy_no_nan", "name": "BACC", "find_best": "max"},
    #         {"metric": "max_val_balanced_binary_accuracy_no_nan", "name": "*Max. BACC*", "find_best": "max",
    #          "is_min_max": True},
    #         {"metric": "max_val_binary_accuracy_no_nan", "name": "*Max. Accuracy*", "find_best": "max",
    #          "is_min_max": True},
    #         {"metric": "max_val_AUC_no_nan", "name": "*Max. AUC*", "find_best": "max", "is_min_max": True}
    #     ]
    # },
    # "ClinToxDataset": {
    #     "general_info": [
    #         "ClinTox (MoleculeNet) consists of 1478 compounds as smiles and ",
    #         "data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons. ",
    #         "We use random 5-fold cross-validation. The first label 'approved' is chosen as target."
    #     ],
    #     "targets": [
    #         {"metric": "val_accuracy", "name": "Accuracy", "find_best": "max"},
    #         {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"},
    #         {"metric": "max_val_accuracy", "name": "*Max. Accuracy*", "find_best": "max", "is_min_max": True},
    #         {"metric": "max_val_auc", "name": "*Max. AUC*", "find_best": "max", "is_min_max": True}
    #     ]
    # },
    # "QM7Dataset": {
    #     "general_info": [
    #         "QM7 dataset is a subset of GDB-13. ",
    #         "Molecules of up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. ",
    #         "We use dataset-specific 5-fold cross-validation. ",
    #         "The atomization energies are given in kcal/mol and are ranging from -800 to -2000 kcal/mol). "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "MAE [kcal/mol]", "find_best": "min"},
    #         {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [kcal/mol]", "find_best": "min"},
    #         {"metric": "min_val_scaled_mean_absolute_error", "name": "*Min. MAE*", "find_best": "min",
    #          "is_min_max": True},
    #         {"metric": "min_val_scaled_root_mean_squared_error", "name": "*Min. RMSE*", "find_best": "min",
    #          "is_min_max": True}
    #     ]
    # },
    # "QM9Dataset": {
    #     "general_info": [
    #         "QM9 dataset of 134k stable small organic molecules made up of C,H,O,N,F. ",
    #         "Labels include geometric, energetic, electronic, and thermodynamic properties. ",
    #         "We use a random 10-fold cross-validation, ",
    #         "but not all splits are evaluated for cheaper evaluation. ",
    #         "Test errors are MAE and for energies are given in [eV]. "
    #     ],
    #     "targets": [
    #         {"metric": "val_scaled_mean_absolute_error", "name": "HOMO [eV]", "find_best": "min",
    #          "multi_target_indices": [5]},
    #         {"metric": "val_scaled_mean_absolute_error", "name": "LUMO [eV]", "find_best": "min",
    #          "multi_target_indices": [6]},
    #         {"metric": "val_scaled_mean_absolute_error", "name": "U0 [eV]", "find_best": "min",
    #          "multi_target_indices": [[10], [15]]},
    #         {"metric": "val_scaled_mean_absolute_error", "name": "H [eV]", "find_best": "min",
    #          "multi_target_indices": [[12], [17]]},
    #         {"metric": "val_scaled_mean_absolute_error", "name": "G [eV]", "find_best": "min",
    #          "multi_target_indices": [[13], [18]]},
    #     ]
    # },
    # "SIDERDataset": {
    #     "general_info": [
    #         "SIDER (MoleculeNet) consists of 1427 compounds as smiles and ",
    #         "data for adverse drug reactions (ADR), grouped into 27 system organ classes. ",
    #         "We use random 5-fold cross-validation."
    #     ],
    #     "targets": [
    #         {"metric": "val_binary_accuracy", "name": "Accuracy", "find_best": "max"},
    #         {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"},
    #         {"metric": "max_val_binary_accuracy", "name": "*Max. Accuracy*", "find_best": "max", "is_min_max": True},
    #         {"metric": "max_val_auc", "name": "*Max. AUC*", "find_best": "max", "is_min_max": True}
    #     ]
    # },
    # "MD17Dataset": {
    #     "general_info": [
    #         "Energies and forces for molecular dynamics trajectories of eight organic molecules. ",
    #         "All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. ",
    #         "We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. ",
    #         "Errors are MAE for forces. ",
    #         "Results are for the CCSD and CCSD(T) data in MD17. "
    #     ],
    #     "targets": [
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Aspirin", "find_best": "min",
    #          "trajectory_name": "aspirin_ccsd"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Toluene", "find_best": "min",
    #          "trajectory_name": "toluene_ccsd_t"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Malonaldehyde", "find_best": "min",
    #          "trajectory_name": "malonaldehyde_ccsd_t"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Benzene", "find_best": "min",
    #          "trajectory_name": "benzene_ccsd_t"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Ethanol", "find_best": "min",
    #          "trajectory_name": "ethanol_ccsd_t"}
    #     ]
    # },
    # "MD17RevisedDataset": {
    #     "general_info": [
    #         "Energies and forces for molecular dynamics trajectories. ",
    #         "All geometries in A, energy labels in kcal/mol and force labels in kcal/mol/A. ",
    #         "We use preset train-test split. Training on 1000 geometries, test on 500/1000 geometries. ",
    #         "Errors are MAE for forces. "
    #     ],
    #     "targets": [
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Aspirin", "find_best": "min",
    #          "trajectory_name": "aspirin"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Toluene", "find_best": "min",
    #          "trajectory_name": "toluene"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Malonaldehyde", "find_best": "min",
    #          "trajectory_name": "malonaldehyde"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Benzene", "find_best": "min",
    #          "trajectory_name": "benzene"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Ethanol", "find_best": "min",
    #          "trajectory_name": "ethanol"}
    #     ]
    # },
    # "ISO17Dataset": {
    #     "general_info": [
    #         "The database consist of 129 molecules each containing 5,000 conformational geometries, ",
    #         "energies and forces with a resolution of 1 femtosecond in the molecular dynamics trajectories. ",
    #         "The molecules were randomly drawn from the largest set of isomers in the QM9 dataset. "
    #     ],
    #     "targets": [
    #         {"metric": "val_energy_scaled_mean_absolute_error", "name": "Energy (test_within)", "find_best": "min"},
    #         {"metric": "val_force_scaled_mean_absolute_error", "name": "Force (test_within)", "find_best": "min"},
    #         {"metric": "min_val_energy_scaled_mean_absolute_error", "name": "*Min. Energy* (test_within)",
    #          "find_best": "min", "is_min_max": True},
    #         {"metric": "min_val_force_scaled_mean_absolute_error", "name": "*Min. Force* (test_within)",
    #          "find_best": "min", "is_min_max": True},
    #     ]
    # },
    # "VgdMockDataset": {
    #     "general_info": [
    #         "Synthetic classification dataset containing 100 small, randomly generated graphs, ",
    #         "where half of them were seeded with a triangular subgraph motif, ",
    #         "which is the explanation ground truth for the target class distinction."
    #     ],
    #     "targets": [
    #         {"metric": "val_categorical_accuracy", "name": "Categorical Accuracy", "find_best": "max"},
    #         {"metric": "val_node_auc", "name": "Node AUC", "find_best": "max"},
    #         {"metric": "val_edge_auc", "name": "Edge AUC", "find_best": "max"},
    #     ]
    # },
    # "VgdRbMotifsDataset": {
    #     "general_info": [
    #         "Synthetic graph regression dataset consisting of 5000 small, randomly generated graphs, ",
    #         "where some of them are seeded with special red- or blue-dominated subgraph motifs, where ",
    #         "blue motifs contribute negatively to a graph's overall target value and red motifs ",
    #         "contribute positively. The explanation ground truth for this datasets consists of these ",
    #         "motifs."
    #     ],
    #     "targets": [
    #         {"metric": "val_mean_squared_error", "name": "MSE", "find_best": "min"},
    #         {"metric": "val_node_auc", "name": "Node AUC", "find_best": "max"},
    #         {"metric": "val_edge_auc", "name": "Edge AUC", "find_best": "max"},
    #     ]
    # }
}


def load_yaml_file(file_path: str):
    with open(file_path, 'r') as stream:
        obj = yaml.safe_load(stream)
    return obj


def make_table_line(tab_list: list):
    return "| " + "".join([str(tab) + " | " for tab in tab_list]) + "\n"


output_file_name = "README%s.md" % ("_min_max" if show_min_max else "")


with open(output_file_name, "w") as f:
    f.write("# Summary of Benchmark Training\n\n")

    f.write("Note that these are the results for models within `kgcnn` implementation, ")
    f.write(
        "and that training is not always done with optimal hyperparameter or splits, when comparing with literature.\n")
    f.write("This table is generated automatically from keras history logs.\n")
    f.write("Model weights and training statistics plots are not uploaded on \n")
    f.write("[github](https://github.com/aimat-lab/gcnn_keras/tree/master/training/results) \n")
    f.write("due to their file size.\n\n")
    f.write("*Max.* or *Min.* denotes the best test error observed for any epoch during training.\n")
    f.write("To show overall best test error run ``python3 summary.py --min_max True``.\n")
    f.write("If not noted otherwise, we use a (fixed) random k-fold split for validation errors.\n\n")

    for dataset, dataset_info in benchmark_datasets.items():
        f.write("#### %s\n\n" % dataset)
        f.write("%s\n\n" % "".join(dataset_info["general_info"]))

        search_path = dataset
        if os.path.exists(search_path):
            model_folders = [f for f in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, f))]
        else:
            raise FileNotFoundError("Dataset results could not be found.")

        model_cols = ["model", "kgcnn", "epochs"]
        for x in dataset_info["targets"]:
            model_cols.append(x["name"])

        df = pd.DataFrame(columns=model_cols)

        for model in model_folders:

            search_path = os.path.join(dataset, model)
            if os.path.exists(search_path):
                output_files = [
                    f for f in os.listdir(search_path) if
                    os.path.isfile(os.path.join(search_path, f)) and f.endswith(".yaml")]
            else:
                print("Model %s could not be found" % model)

            result_dict = {}
            for r in output_files:
                results = load_yaml_file(os.path.join(dataset, model, r))
                result_dict.update({
                    "model": results["model_name"] if results["model_class"] == "make_model" else "%s.%s" % (
                        results["model_name"], results["model_class"]),
                    "kgcnn": results["kgcnn_version"],
                    "epochs": str(int(np.mean(results["epochs"]))),
                })
                for x in dataset_info["targets"]:
                    if x["metric"] not in results:
                        continue
                    target_res = np.array(results[x["metric"]])
                    if "multi_target_indices" in results:
                        target_idx = results["multi_target_indices"]
                        if target_idx is not None:
                            if "multi_target_indices" not in x:
                                continue
                            info_idx = x["multi_target_indices"]
                            if not isinstance(info_idx[0], list):
                                info_idx = [info_idx]  # Make list of list.
                            if target_idx not in info_idx:
                                continue
                    if "trajectory_name" in results:
                        target_idx = results["trajectory_name"]
                        if target_idx is not None:
                            if "trajectory_name" not in x:
                                continue
                            info_idx = x["trajectory_name"]
                            if not isinstance(info_idx[0], list):
                                info_idx = [info_idx]  # Make list of list.
                            if target_idx not in info_idx:
                                continue
                    target_res = target_res[~np.isnan(target_res)]
                    if len(target_res) <= 1:
                        result_dict[x["name"]] = (np.mean(target_res), nan)
                    else:
                        result_dict[x["name"]] = (np.mean(target_res), np.std(target_res))
            df = pd.concat([df, pd.DataFrame({key: [value] for key, value in result_dict.items()})])

        # Pandas style does not seem to support mark-down formatting.
        # Manual formatting here.
        for data_targets in dataset_info["targets"]:
            target_val = df[data_targets["name"]]
            find_function = np.argmax if data_targets["find_best"] == "max" else np.argmin
            best = find_function([x[0] if isinstance(x, (list, tuple, np.ndarray)) else x for x in target_val])
            format_strings = ["{0:0.4f} &pm; {1:0.4f}"] * len(target_val)
            format_strings[int(best)] = "**{0:0.4f} &pm; {1:0.4f}**"
            format_val = [
                format_strings[i].format(*v) if isinstance(v, (list, tuple, np.ndarray)) else format_strings[i].format(
                    v, nan) for i, v in enumerate(target_val)]
            df[data_targets["name"]] = format_val

            if not show_min_max:
                if "is_min_max" in data_targets:
                    if data_targets["is_min_max"]:
                        del df[data_targets["name"]]

        f.write(df.to_markdown(index=False))
        f.write("\n\n")
