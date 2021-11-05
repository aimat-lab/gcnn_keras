import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from tensorflow_addons import optimizers
from kgcnn.utils import learning
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.data.datasets.ESOL import ESOLDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelection
from kgcnn.utils.data import save_json_file
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on ESOL dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="AttentiveFP")  # AttentiveFP
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.",
                    default="hyper/hyper_esol.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification
model_name = args["model"]
model_selection = ModelSelection()
make_model = model_selection.make_model(model_name)

# Hyper-parameter.
hyper_selection = HyperSelection(args["hyper"], model_name=model_name)
hyper = hyper_selection.hyper()

# Loading ESOL Dataset
hyper_data = hyper['data']
dataset = ESOLDataset().set_attributes()
if "set_range" in hyper_data:
    dataset.set_range(**hyper_data['set_range'])
if "set_edge_indices_reverse" in hyper_data:
    dataset.set_edge_indices_reverse()
data_name = dataset.dataset_name
data_unit = "mol/L"
data_length = dataset.length

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
data_loader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = np.array(dataset.graph_labels)

# Test Split
kf = KFold(**hyper_selection.k_fold())
split_indices = kf.split(X=np.arange(data_length-1)[:, None])

# Variables
history_list, test_indices_list = [], []
model, scaler, xtest, ytest = None, None, None, None

# Training on splits
for train_index, test_index in split_indices:

    # Make model.
    model = make_model(**hyper['model'])

    # Select train and test data.
    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = data_loader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = data_loader[test_index].tensor(ragged=is_ragged), labels[test_index]

    # Normalize training and test targets.
    scaler = StandardScaler(with_std=True, with_mean=True, copy=True)
    ytrain = scaler.fit_transform(ytrain)
    ytest = scaler.transform(ytest)

    # Get optimizer from serialized hyper-parameter.
    mae_metric = ScaledMeanAbsoluteError((1, 1), name="mean_absolute_error")
    rms_metric = ScaledRootMeanSquaredError((1, 1))
    if scaler.scale_ is not None:
        mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
        rms_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
    model.compile(**hyper_selection.compile(loss='mean_squared_error', metrics=[mae_metric, rms_metric]))
    print(model.summary())

    # Start and time training
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtest, ytest),
                     **hyper_selection.fit()
                     )
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directory
hyper_info = deepcopy(hyper["info"])
post_fix = str(hyper_info["postfix"]) if "postfix" in hyper_info else ""
post_fix_file = str(hyper_info["postfix_file"]) if "postfix_file" in hyper_info else ""
os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("results", data_name), exist_ok=True)
filepath = os.path.join("results", data_name, hyper['model']['name'] + post_fix)
os.makedirs(filepath, exist_ok=True)

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name="mean_absolute_error", val_loss_name="val_mean_absolute_error",
                     model_name=model_name, data_unit=data_unit, dataset_name=data_name, filepath=filepath,
                     file_name="mae_esol" + post_fix_file + ".png")
# Plot prediction
plot_predict_true(scaler.inverse_transform(model.predict(xtest)), scaler.inverse_transform(ytest), filepath=filepath,
                  model_name=model_name, dataset_name=data_name, file_name="predict_esol" + post_fix_file + ".png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + post_fix_file + ".npz"), test_indices_list)

# Save hyper-parameter again, which were used for this fit.
save_json_file(hyper, os.path.join(filepath, model_name + "_hyper" + post_fix_file + ".json"))
