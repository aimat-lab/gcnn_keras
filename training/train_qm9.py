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

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on QM9 dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="Megnet")
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
data_name = dataset.dataset_name
data_length = dataset.length
target_names = dataset.target_names

# Prepare actual training data.
data_points_to_use = hyper_data['data_points_to_use'] if "data_points_to_use" in hyper_data else data_length
target_indices = np.array(hyper_data['target_indices'], dtype="int")
target_unit_conversion = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 27.2114, 27.2114, 27.2114, 1.0, 27.2114, 27.2114, 27.2114,
                                    27.2114, 27.2114, 1.0]])  # Pick always same units for training
data_unit = ["GHz", "GHz", "GHz", "D", r"a_0^3", "eV", "eV", "eV", r"a_0^2", "eV", "eV", "eV", "eV", "eV", r"cal/mol K"]
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

train_loss = []
test_loss = []
mae_5fold = []
all_test_index = []
splits_done = 0
model, scaler, xtest, ytest, mae_valid, atoms_test = None, None, None, None, None, None
for train_index, test_index in split_indices:
    # Only do execute_splits out of the 10-folds
    if splits_done >= execute_splits:
        break
    # Make model.
    model = make_model(**hyper['model'])

    # Select train and test data.
    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain = dataloader[train_index].tensor(ragged=is_ragged)
    xtest = dataloader[test_index].tensor(ragged=is_ragged)
    atoms_test = [atoms[i] for i in test_index]
    atoms_train = [atoms[i] for i in train_index]
    labels_train = labels[train_index]
    labels_test = labels[test_index]

    # Normalize training and test targets.
    scaler = QM9GraphLabelScaler().fit(atoms_train, labels_train)
    ytrain = scaler.fit_transform(atoms_train, labels_train)[:, target_indices]
    ytest = scaler.transform(atoms_test, labels_test)[:, target_indices]

    # Compile Metrics and loss. Use a scaled metric to logg the unscaled targets in fit().
    std_scale = np.expand_dims(scaler.scale_[target_indices], axis=0)
    mae_metric = ScaledMeanAbsoluteError(std_scale.shape)
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
    train_mae = np.array(hist.history['mean_absolute_error'])
    train_loss.append(train_mae)
    val_mae = np.array(hist.history['val_mean_absolute_error'])
    test_loss.append(val_mae)
    mae_valid = np.mean(val_mae[-5:], axis=0)
    mae_5fold.append(mae_valid)
    all_test_index.append([data_selection[train_index], data_selection[test_index]])
    splits_done += 1

# Make output directories
hyper_info = deepcopy(hyper["info"])
post_fix = str(hyper_info["postfix"]) if "postfix" in hyper_info else ""
post_fix_file = str(hyper_info["postfix_file"]) if "postfix_file" in hyper_info else ""
os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("results", data_name), exist_ok=True)
filepath = os.path.join("results", data_name, hyper['model']['name'] + post_fix)
os.makedirs(filepath, exist_ok=True)

# Plot training- and test-loss vs epochs for all splits.
plt.figure()
for x in train_loss:
    plt.plot(np.arange(x.shape[0]), x, c='red', alpha=0.85)
for y in test_loss:
    plt.plot((np.arange(len(y)) + 1) * epostep, y, c='blue', alpha=0.85)
plt.scatter([train_loss[-1].shape[0]], [np.mean(mae_5fold)],
            label=r"Test: {0:0.4f} $\pm$ {1:0.4f} ".format(np.mean(mae_5fold), np.std(mae_5fold)), c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('QM9 Loss' + model_name)
plt.legend(loc='upper right', fontsize='medium')
plt.savefig(os.path.join(filepath, model_name + "_mae_qm9" + post_fix_file + ".png"))
plt.show()

# Plot predicted targets vs actual targets for last split. This will be adapted for all targets in the future.
true_test = scaler.inverse_transform(atoms_test, scaler.padd(ytest, target_indices))[:, target_indices]
pred_test = scaler.inverse_transform(atoms_test, scaler.padd(model.predict(xtest), target_indices))[:, target_indices]
mae_last = np.mean(np.abs(true_test - pred_test), axis=0)
plt.figure()
for i, ti in enumerate(target_indices):
    plt.scatter(pred_test[:, i], true_test[:, i], alpha=0.3,
                label=target_names[ti] + " MAE: {0:0.4f} ".format(mae_last[i]) + "[" + data_unit[ti] + "]")
plt.plot(np.arange(np.amin(true_test), np.amax(true_test), 0.05),
         np.arange(np.amin(true_test), np.amax(true_test), 0.05), color='red')
plt.xlabel('Predicted Last Split')
plt.ylabel('Actual')
plt.legend(loc='upper left', fontsize='x-small')
plt.title("Prediction Test for " + str(model_name))
plt.savefig(os.path.join(filepath, model_name + "_predict_qm9" + post_fix_file + ".png"))
plt.show()

# Save hyper-parameter again, which were used for this fit.
save_json_file(hyper, os.path.join(filepath, model_name + "_hyper" + post_fix_file + ".json"))

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + post_fix_file + ".npz"), all_test_index)
