import logging
import numpy as np
import argparse
import os
import time

from datetime import timedelta
from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.selection.models import ModelSelection
from kgcnn.selection.data import DatasetSelection
from kgcnn.selection.hyper import HyperSelection
from tensorflow_addons import optimizers
from kgcnn.utils import learning
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true

# Input arguments from command line.
parser = argparse.ArgumentParser(description='Train a GNN on a Molecule dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="GCN")
parser.add_argument("--dataset", required=False, help="Name of the dataset or leave empty for custom dataset.",
                    default="ESOLDataset")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_esol.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Main parameter about model, dataset, and hyper-parameter
model_name = args["model"]
dataset_name = args["dataset"]
hyper_path = args["hyper"]

# HyperSelection is used to store and verify hyper-parameters.
hyper = HyperSelection(hyper_path, model_name=model_name, dataset_name=dataset_name)

# Model Selection to load a model definition from a module in kgcnn.literature
model_selection = ModelSelection(model_name)
make_model = model_selection.make_model()

# Loading a dataset from a module in kgcnn.data.datasets.
# If no name is given, a general MoleculeNetDataset() is constructed.
# However, the construction then must be fully defined in the data section of the hyper-parameters,
# including all methods to run on the dataset.
data_selection = DatasetSelection(dataset_name)

# Make dataset.
try:
    dataset = data_selection.dataset(**hyper.data("dataset"))
except NotImplementedError:
    print("ERROR: Dataset not found, try general `MoleculeNetDataset`...")
    dataset = MoleculeNetDataset(**hyper.data("dataset"))

# Set methods on the dataset to apply encoders or transformations or reload the data with different parameters.
# This is only done, if there is a entry with functional kwargs in hyper-parameters.
data_selection.perform_methods_on_dataset(
    dataset, ["prepare_data", "read_in_memory", "set_attributes", "set_range", "set_angle",
              "normalize_edge_weights_sym", "set_edge_indices_reverse"], hyper.data())
data_selection.assert_valid_model_input(dataset, hyper.inputs())
dataset.clean(hyper.inputs())
data_length = len(dataset)

# For MoleculeNetDataset, always train on graph, labels.
labels = np.array(dataset.graph_labels)
if len(labels.shape) <= 1:
    labels = np.expand_dims(labels, axis=-1)

# Training on splits
kf = KFold(**hyper.cross_validation())
history_list, test_indices_list, model, hist = [], [], None, None
for train_index, test_index in kf.split(X=np.arange(data_length)[:, None]):

    # Select train and test data.
    xtrain, ytrain = dataset[train_index].tensor(hyper.inputs()), labels[train_index]
    xtest, ytest = dataset[test_index].tensor(hyper.inputs()), labels[test_index]

    # Normalize training and test targets.
    if hyper.use_scaler():
        print("INFO: Using StandardScaler.")
        scaler = StandardScaler(**hyper.scaler())
        ytrain = scaler.fit_transform(ytrain)
        ytest = scaler.transform(ytest)

        # If scaler was used we add rescaled standard metrics to loss
        mae_metric = ScaledMeanAbsoluteError((1, 1), name="scaled_mean_absolute_error")
        rms_metric = ScaledRootMeanSquaredError((1, 1), name="scaled_root_mean_squared_error")
        if scaler.scale_ is not None:
            mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
            rms_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
        metrics = [mae_metric, rms_metric]
    else:
        print("INFO: Not using StandardScaler.")
        metrics = None

    # Make the model for current split.
    model = make_model(**hyper.make_model())

    # Compile model with optimizer and loss
    model.compile(**hyper.compile(metrics=metrics))
    print(model.summary())

    # Start and time training
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtest, ytest),
                     **hyper.fit()
                     )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=model_name, data_unit="", dataset_name=dataset_name,
                     filepath=filepath, file_name="loss" + postfix_file + ".png")
# Plot prediction
predicted_y = model.predict(xtest)
true_y = ytest

if hyper.use_scaler():
    predicted_y = scaler.inverse_transform(predicted_y)
    true_y = scaler.inverse_transform(true_y)

plot_predict_true(predicted_y, true_y,
                  filepath=filepath, data_unit=hyper.data_unit(),
                  model_name=model_name, dataset_name=dataset_name,
                  file_name="predict" + postfix_file + ".png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

# Save hyper-parameter again, which were used for this fit.
hyper.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))
