import os
import numpy as np
import pandas as pd
from kgcnn.data.utils import load_yaml_file

benchmark_datasets = {
    "CoraLuDataset": {
        "general_info": [
            "Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse attributes and 7 node classes. ",
            "Here we use random 5-fold cross-validation on nodes. "
        ],
        "targets": [
            {"metric": "val_categorical_accuracy", "name": "Categorical accuracy", "find_best": "max"}
        ]
    },
    "CoraDataset": {
        "general_info": [
            "Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. ",
            "Here we use random 5-fold cross-validation on nodes. "
        ],
        "targets": [
            {"metric": "val_categorical_accuracy", "name": "Categorical accuracy", "find_best": "max"}
        ]
    },
    "ESOLDataset": {
        "general_info": [
            "ESOL consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). ",
            "We use random 5-fold cross-validation. "
        ],
        "targets": [
            {"metric": "val_scaled_mean_absolute_error", "name": "MAE [log mol/L]", "find_best": "min"},
            {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log mol/L]", "find_best": "min"}
        ]
    },
    "LipopDataset": {
        "general_info": [
            "Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles. ",
            "Graph labels for regression are octanol/water distribution coefficient (logD at pH 7.4). ",
            "We use random 5-fold cross-validation. "
        ],
        "targets": [
            {"metric": "val_scaled_mean_absolute_error", "name": "MAE [log mol/L]", "find_best": "min"},
            {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log mol/L]", "find_best": "min"}
        ]
    },
    "MatProjectEFormDataset": {
        "general_info": [
            "Materials Project dataset from Matbench with 132752 crystal structures ",
            "and their corresponding formation energy in [eV/atom]. ",
            "We use a random 10-fold cross-validation. "
        ],
        "targets": [
            {"metric": "val_scaled_mean_absolute_error", "name": "MAE [eV/atom]", "find_best": "min"},
            {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [eV/atom]", "find_best": "min"}
        ]
    },
    "MutagenicityDataset": {
        "general_info": [
            "Mutagenicity dataset from TUDataset for classification with 4337 graphs. ",
            "The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation. "
        ],
        "targets": [
            {"metric": "val_accuracy", "name": "Accuracy", "find_best": "max"},
            {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"}
        ]
    },
    "MUTAGDataset": {
        "general_info": [
            "MUTAG dataset from TUDataset for classification with 188 graphs. ",
            "We use random 5-fold cross-validation. "
        ],
        "targets": [
            {"metric": "val_accuracy", "name": "Accuracy", "find_best": "max"},
            {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"}
        ]
    },
    "FreeSolvDataset": {
        "general_info": [
            "FreeSolv (MoleculeNet) consists of 642 compounds as smiles and ",
            "their corresponding hydration free energy for small neutral molecules in water. ",
            "We use a random 5-fold cross-validation. ",
            "`Min. RMSE` denotes the smallest test RMSE observed for any epoch. "
        ],
        "targets": [
            {"metric": "val_scaled_mean_absolute_error", "name": "MAE [log mol/L]", "find_best": "min"},
            {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log mol/L]", "find_best": "min"},
            {"metric": "min_val_scaled_root_mean_squared_error", "name": "Min. RMSE [log mol/L]", "find_best": "min"}
        ]
    },
    "PROTEINSDataset": {
        "general_info": [
            "TUDataset of proteins that are classified as enzymes or non-enzymes. ",
            "Nodes represent the amino acids of the protein. ",
            "We use random 5-fold cross-validation. "
        ],
        "targets": [
            {"metric": "val_accuracy", "name": "Accuracy", "find_best": "max"},
            {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"}
        ]
    },
    "Tox21MolNetDataset": {
        "general_info": [
            "Tox21 (MoleculeNet) consists of 7831 compounds as smiles and ",
            "12 different targets relevant to drug toxicity. ",
            "We use random 5-fold cross-validation. "
        ],
        "targets": [
            {"metric": "val_binary_accuracy", "name": "Accuracy", "find_best": "max"},
            {"metric": "val_auc", "name": "AUC(ROC)", "find_best": "max"}
        ]
    },
    "QM7Dataset": {
        "general_info": [
            "QM7 dataset is a subset of GDB-13. ",
            "Molecules of up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. ",
            "The atomization energies are given in kcal/mol and are ranging from -800 to -2000 kcal/mol). "
        ],
        "targets": [
            {"metric": "val_scaled_mean_absolute_error", "name": "MAE [kcal/mol]", "find_best": "min"},
            {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [kcal/mol]", "find_best": "min"}
        ]
    },
}


def make_table_line(tab_list: list):
    return "| " + "".join([str(tab) + " | " for tab in tab_list]) + "\n"


with open("README.md", "w") as f:
    f.write("# Summary of Benchmark Training\n\n")

    f.write("Note that these are the results for models within `kgcnn` implementation, ")
    f.write(
        "and that training is not always done with optimal hyperparameter or splits, when comparing with literature.\n")
    f.write("This table is generated automatically from keras history logs.\n")
    f.write("Model weights and training statistics plots are not uploaded on github due to their file size.\n\n")

    for dataset, dataset_info in benchmark_datasets.items():
        f.write("## %s\n\n" % dataset)
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
                    target_res = np.array(results[x["metric"]])
                    target_res = target_res[~np.isnan(target_res)]
                    result_dict[x["name"]] = (np.mean(target_res), np.std(target_res))
            df = pd.concat([df, pd.DataFrame({key: [value] for key, value in result_dict.items()})])

        # Pandas style does not seem to support mark-down formatting.
        # Manual formatting here.
        for data_targets in dataset_info["targets"]:
            target_val = df[data_targets["name"]]
            find_function = np.argmax if data_targets["find_best"] == "max" else np.argmin
            best = find_function([x[0] for x in target_val])
            format_strings = ["{0:0.4f} &pm; {1:0.4f}"] * len(target_val)
            format_strings[int(best)] = "**{0:0.4f} &pm; {1:0.4f}**"
            format_val = [format_strings[i].format(x, y) for i, (x, y) in enumerate(target_val)]
            df[data_targets["name"]] = format_val

        f.write(df.to_markdown(index=False))
        f.write("\n\n")
