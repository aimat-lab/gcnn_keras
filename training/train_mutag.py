import numpy as np
import argparse
import os

from kgcnn.data.datasets.mutag import MUTAGDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelection
from kgcnn.training.graph import train_graph_classification_supervised
from sklearn.model_selection import KFold
from kgcnn.utils.plots import plot_train_test_loss

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Mutagenicity dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="INorp")  # INorp
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.",
                    default="hyper/hyper_mutag.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification
model_name = args["model"]
ms = ModelSelection()
make_model = ms.make_model(model_name)

# Find hyper-parameter.
hyper_selection = HyperSelection(args["hyper"], model_name=model_name, dataset_name="MUTAG")
hyper = hyper_selection.get_hyper()

# Loading MUTAG Dataset
hyper_data = hyper['data']
dataset = MUTAGDataset()
dataset_name = dataset.dataset_name
data_length = dataset.length
if "set_edge_indices_reverse" in hyper_data:
    dataset.set_edge_indices_reverse()

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
data_loader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = np.expand_dims(dataset.graph_labels, axis=-1)

# Test Split
kf = KFold(**hyper_selection.k_fold())

# Train on splits
history_list, test_indices_list = [], []
for train_index, test_index in kf.split(X=np.arange(data_length)[:, None]):
    # Select train and test data.
    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = data_loader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = data_loader[test_index].tensor(ragged=is_ragged), labels[test_index]

    # Use a generic training function for graph classification.
    model, hist = train_graph_classification_supervised(xtrain, ytrain,
                                                        validation_data=(xtest, ytest),
                                                        make_model=make_model,
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
