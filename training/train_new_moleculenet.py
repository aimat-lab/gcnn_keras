import numpy as np
import time
import os
import argparse

from sklearn.preprocessing import StandardScaler
from tensorflow_addons import optimizers
from kgcnn.utils import learning
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelection
from kgcnn.utils.data import save_json_file
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.training.graph import train_graph_regression_supervised

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Molecule dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="DMPNN")  # AttentiveFP
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.",
                    default="hyper/hyper_new_moleculenet.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification
model_name = args["model"]
model_selection = ModelSelection()
make_model = model_selection.make_model(model_name)

# Hyper-parameter.
hyper_selection = HyperSelection(args["hyper"], model_name=model_name, dataset_name="Example")
hyper = hyper_selection.get_hyper()

# Loading ESOL Dataset
hyper_data = hyper['data']
dataset = MoleculeNetDataset(**hyper_data["dataset"])
for method_data in ["prepare_data", "read_in_memory", "set_attributes", "set_range", "set_edge_indices_reverse",
                    "set_angle"]:
    if hasattr(dataset, method_data) and method_data in hyper_data:
        getattr(dataset, method_data)(**hyper_data[method_data])
dataset_name = dataset.dataset_name
data_unit = ""
data_length = dataset.length

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = np.array(dataset.graph_labels)

# Define Scaler for targets.
scaler = StandardScaler(with_std=True, with_mean=True, copy=True)

# Test Split
kf = KFold(**hyper_selection.k_fold())

# Training on splits
history_list, test_indices_list = [], []
for train_index, test_index in kf.split(X=np.arange(data_length)[:, None]):
    # Select train and test data.
    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = dataloader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = dataloader[test_index].tensor(ragged=is_ragged), labels[test_index]

    # Normalize training and test targets.
    scaler = StandardScaler(with_std=True, with_mean=True, copy=True)
    ytrain = scaler.fit_transform(ytrain)
    ytest = scaler.transform(ytest)

    # Use a generic training function for graph regression.
    model, hist = train_graph_regression_supervised(xtrain, ytrain,
                                                    validation_data=(xtest, ytest),
                                                    make_model=make_model,
                                                    hyper_selection=hyper_selection,
                                                    scaler=scaler)

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directory
filepath = hyper_selection.results_file_path()
postfix_file = hyper_selection.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name="mean_absolute_error", val_loss_name="val_mean_absolute_error",
                     model_name=model_name, data_unit=data_unit, dataset_name=dataset_name, filepath=filepath,
                     file_name="mae" + postfix_file + ".png")
# Plot prediction
plot_predict_true(scaler.inverse_transform(model.predict(xtest)), scaler.inverse_transform(ytest), filepath=filepath,
                  model_name=model_name, dataset_name=dataset_name, file_name="predict" + postfix_file + ".png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

# Save hyper-parameter again, which were used for this fit.
save_json_file(hyper, os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))
