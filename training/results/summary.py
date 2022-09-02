import os
import numpy as np
from kgcnn.data.utils import load_yaml_file


benchmark_datasets = {
    "CoraLuDataset": {
        "general_info": "Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse node attributes and 7 node classes. Here we use random 5-fold cross-validation  on nodes.",
        "targets": [{"metric": "val_categorical_accuracy", "name": "Categorical accuracy"}]
    },
    "CoraDataset": {
        "general_info": "Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes. Here we use random 5-fold cross-validation on nodes.",
        "targets": [{"metric": "val_categorical_accuracy", "name": "Categorical accuracy"}]
    },
    "ESOLDataset": {
        "general_info": "ESOL (MoleculeNet) consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). We use random 5-fold cross-validation.",
        "targets": [{"metric": "val_scaled_mean_absolute_error", "name": "MAE [log mol/L]"},
                    {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log mol/L]"}]
    },
    "LipopDataset": {
        "general_info": "Lipophilicity (MoleculeNet) consists of 4200 compounds as smiles and their corresponding octanol/water distribution coefficient (logD at pH 7.4). We use random 5-fold cross-validation.",
        "targets": [{"metric": "val_scaled_mean_absolute_error", "name": "MAE [log mol/L]"},
                    {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [log mol/L]"}]
    },
    "MatProjectEFormDataset": {
        "general_info": "Materials Project dataset from Matbench with 132752 crystal structures and their corresponding formation energy in [eV/atom]. We use random 10-fold cross-validation.",
        "targets": [{"metric": "val_scaled_mean_absolute_error", "name": "MAE [eV/atom]"},
                    {"metric": "val_scaled_root_mean_squared_error", "name": "RMSE [eV/atom]"}]
    },
    "MutagenicityDataset": {
        "general_info": "Mutagenicity dataset from TUDataset for classification with 4337 graphs. The dataset was cleaned for unconnected atoms. We use random 5-fold cross-validation.",
        "targets": [{"metric": "val_accuracy", "name": "Accuracy"},
                    {"metric": "val_auc", "name": "AUC(ROC)"}]
    },
    "MUTAGDataset": {
        "general_info": "MUTAG dataset from TUDataset for classification with 188 graphs. We use random 5-fold cross-validation.",
        "targets": [{"metric": "val_accuracy", "name": "Accuracy"},
                    {"metric": "val_auc", "name": "AUC(ROC)"}]
    }
}


def make_table_line(tab_list: list):
    return "| " + "".join([str(tab) + " | " for tab in tab_list]) + "\n"


with open("README.md", "w") as f:
    f.write("# Summary of Benchmark Training\n\n")

    f.write("Note that these are the results for models within `kgcnn` implementation, ")
    f.write("and that training is not always done with optimal hyperparameter or splits, when comparing with literature.\n")
    f.write("This table is generated automatically from keras history logs.\n")
    f.write("Model weights and training statistics plots are not uploaded on github due to their file size.\n\n")

    for dataset, dataset_info in benchmark_datasets.items():
        f.write("## %s\n\n" % dataset)
        f.write("%s\n\n" % dataset_info["general_info"])

        search_path = dataset
        if os.path.exists(search_path):
            model_folders = [f for f in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, f))]
        else:
            raise FileNotFoundError("Dataset results could not be found.")

        model_cols = ["model", "kgcnn", "epochs"]
        for x in dataset_info["targets"]:
            model_cols.append(x["name"])
        f.write(make_table_line(model_cols))
        f.write(make_table_line([":---:"]*len(model_cols)))

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
                    result_dict[x["name"]] = "{0:0.4f} &pm; {1:0.4f} ".format(
                        np.mean(results[x["metric"]]), np.std(results[x["metric"]]))

            f.write(make_table_line([result_dict[x] if x in result_dict else "" for x in model_cols]))

        f.write("\n")
