import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import kgcnn.utils.learning
import argparse

from copy import deepcopy
from tensorflow_addons import optimizers
from sklearn.preprocessing import StandardScaler
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from kgcnn.data.datasets.qm9 import QM9Dataset, QM9GraphLabelScaler
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.utils.data import save_json_file
from kgcnn.hyper.selection import HyperSelectionTraining
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on QM9 dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="PAiNN")  # Schnet
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.", default="hyper/hyper_qm9.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification
model_name = args["model"]
ms = ModelSelection()
make_model = ms.make_model(model_name)

# Hyper-parameter identification
hyper_selection = HyperSelectionTraining(args["hyper"], model_name=model_name)
hyper = hyper_selection.get_hyper()

# Loading QM9 Dataset
hyper_data = hyper['data']
dataset = QM9Dataset()
# Modifications to set range and angle indices.
if "set_range" in hyper_data:
    dataset.set_range(**hyper_data['set_range'])
if "set_angle" in hyper_data or "requires_angles" in hyper_data:
    dataset.set_angle()
if "set_edge_indices_reverse" in hyper_data:
    dataset.set_edge_indices_reverse()
dataset_name = dataset.dataset_name
data_length = dataset.length
target_names = dataset.target_names
target_unit_conversion = dataset.target_unit_conversion
data_unit = dataset.target_units

# Prepare actual training data.
data_points_to_use = hyper_data['data_points_to_use'] if "data_points_to_use" in hyper_data else data_length
target_indices = np.array(hyper_data['target_indices'], dtype="int")
data_selection = shuffle(np.arange(data_length))[:data_points_to_use]

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])[data_selection]
labels = dataset.graph_labels[data_selection] * target_unit_conversion
atoms = [dataset.node_number[i] for i in data_selection]

# Data-set split
k_fold_info = hyper["training"]["KFold"]
execute_splits = hyper["training"]['execute_folds']  # All splits may be too expensive for QM9
# For validation, use a KFold() split.
kf = KFold(**k_fold_info)
split_indices = kf.split(X=np.arange(len(labels))[:, None])

# hyper_fit and epochs
hyper_fit = hyper_selection.fit()
epo = hyper_fit['epochs']
epostep = hyper_fit['validation_freq']
batch_size = hyper_fit['batch_size']

# Training on splits
splits_done = 0
history_list, test_indices_list = [], []
for train_index, test_index in split_indices:
    # Only do execute_splits out of the 10-folds
    if splits_done >= execute_splits:
        break

    # Select train and test data.
    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain = dataloader[train_index].tensor(ragged=is_ragged)
    xtest = dataloader[test_index].tensor(ragged=is_ragged)
    atoms_test = [atoms[i] for i in test_index]
    atoms_train = [atoms[i] for i in train_index]
    labels_train = labels[train_index]
    labels_test = labels[test_index]

    # Make model.
    model = make_model(**hyper['model'])

    # Normalize training and test targets.
    scaler = QM9GraphLabelScaler().fit(atoms_train, labels_train)
    ytrain = scaler.fit_transform(atoms_train, labels_train)[:, target_indices]
    ytest = scaler.transform(atoms_test, labels_test)[:, target_indices]

    # Compile Metrics and loss. Use a scaled metric to logg the unscaled targets in fit().
    std_scale = np.expand_dims(scaler.scale_[target_indices], axis=0)
    mae_metric = ScaledMeanAbsoluteError(std_scale.shape, name="mean_absolute_error")
    rms_metric = ScaledRootMeanSquaredError(std_scale.shape)
    if scaler.scale_ is not None:
        mae_metric.set_scale(std_scale)
        rms_metric.set_scale(std_scale)
    hyper_compile = hyper_selection.compile(loss='mean_absolute_error', metrics=[mae_metric, rms_metric])
    model.compile(**hyper_compile)
    print(model.summary())

    # Start and time training
    hyper_fit = hyper_selection.fit()
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtest, ytest),
                     **hyper_fit
                     )
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directories
filepath = hyper_selection.results_file_path()
postfix_file = hyper_selection.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name="mean_absolute_error", val_loss_name="val_mean_absolute_error",
                     model_name=model_name, data_unit="", dataset_name=dataset_name, filepath=filepath,
                     file_name="MAE" + postfix_file + ".png")

# Plot predicted targets vs actual targets for last split. This will be adapted for all targets in the future.
true_test = scaler.inverse_transform(atoms_test, scaler.padd(ytest, target_indices))[:, target_indices]
pred_test = scaler.inverse_transform(atoms_test, scaler.padd(model.predict(xtest), target_indices))[:, target_indices]

plot_predict_true(pred_test, true_test, filepath=filepath, data_unit=[data_unit[x] for x in target_indices],
                  model_name=model_name, dataset_name=dataset_name,
                  target_names=[target_names[x] for x in target_indices], file_name="predict" + postfix_file + ".png")

# Save hyper-parameter again, which were used for this fit.
save_json_file(hyper, os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)
