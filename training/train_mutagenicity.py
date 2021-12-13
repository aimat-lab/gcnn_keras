import numpy as np
import argparse
import os

from kgcnn.data.datasets.mutagenicity import MutagenicityDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelection
from kgcnn.training.graph import train_graph_classification_supervised
from sklearn.model_selection import KFold
from kgcnn.utils.plots import plot_train_test_loss

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Mutagenicity dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="GraphSAGE")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.",
                    default="hyper/hyper_mutagenicity.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)
model_name = args["model"]
dataset_name = "Mutagenicity"
hyper_path = args["hyper"]

# Model
model_selection = ModelSelection(model_name)

# Hyper-parameter.
hyper_selection = HyperSelection(hyper_path, model_name=model_name, dataset_name=dataset_name)

# Loading MUTAG Dataset
dataset = MutagenicityDataset()
dataset.hyper_set_graph_methods(hyper_selection.data())
dataset.hyper_assert_valid_model_input(hyper_selection.inputs())
data_length = len(dataset)
labels = np.array(dataset.graph_labels)

# Test Split
kf = KFold(**hyper_selection.k_fold())

# Train on splits
history_list, test_indices_list = [], []
for train_index, test_index in kf.split(X=np.arange(data_length)[:, None]):
    # Select train and test data.
    xtrain, ytrain = dataset[train_index].tensor(hyper_selection.inputs()), labels[train_index]
    xtest, ytest = dataset[test_index].tensor(hyper_selection.inputs()), labels[test_index]

    # Use a generic training function for graph classification.
    model, hist = train_graph_classification_supervised(xtrain, ytrain,
                                                        validation_data=(xtest, ytest),
                                                        make_model=model_selection,
                                                        hyper_selection=hyper_selection,
                                                        metrics=["accuracy"])

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directories.
filepath = hyper_selection.results_file_path()
postfix_file = hyper_selection.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name="accuracy", val_loss_name="val_accuracy",
                     model_name=model_name, data_unit="", dataset_name=dataset_name, filepath=filepath,
                     file_name="acc" + postfix_file + ".png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

# Save hyper-parameter again, which were used for this fit.
hyper_selection.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))
