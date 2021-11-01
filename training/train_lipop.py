import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from kgcnn.utils import learning
from tensorflow_addons import optimizers
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.data.datasets.lipop import LipopDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.data import save_json_file
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelectionTraining

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Lipop dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="AttentiveFP")  # AttentiveFP
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.", default="hyper/hyper_lipop.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model
model_name = args["model"]
ms = ModelSelection()
make_model = ms.make_model(model_name)

# Hyper-parameter.
hyper_selection = HyperSelectionTraining(args["hyper"], model_name=model_name)
hyper = hyper_selection.get_hyper()

# Loading Lipop Dataset
hyper_data = hyper['data']
dataset = LipopDataset().set_attributes()
if "set_range" in hyper_data:
    dataset.set_range(**hyper_data["set_range"])
if "set_edge_indices_reverse" in hyper_data:
    dataset.set_edge_indices_reverse()
data_name = dataset.dataset_name
data_unit = "logD at pH 7.4"
data_length = dataset.length
k_fold_info = hyper["training"]["KFold"]

# Data-set split
kf = KFold(**k_fold_info)
split_indices = kf.split(X=np.arange(data_length)[:, None])

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = np.array(dataset.graph_labels)

# hyper_fit and epochs
hyper_fit = hyper_selection.fit()
epo = hyper_fit['epochs']
epostep = hyper_fit['validation_freq']
batch_size = hyper_fit['batch_size']

train_loss = []
test_loss = []
mae_5fold = []
all_test_index = []
model, scaler, xtest, ytest, mae_valid = None, None, None, None, None
for train_index, test_index in split_indices:
    # Make model.
    model = make_model(**hyper['model'])

    # Select train and test data.
    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = dataloader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = dataloader[test_index].tensor(ragged=is_ragged), labels[test_index]

    # Normalize training and test targets.
    scaler = StandardScaler(with_std=True, with_mean=True, copy=True)
    ytrain = scaler.fit_transform(ytrain)
    ytest = scaler.transform(ytest)

    # Get optimizer from serialized hyper-parameter.
    mae_metric = ScaledMeanAbsoluteError((1, 1))
    rms_metric = ScaledRootMeanSquaredError((1, 1))
    if scaler.scale_ is not None:
        mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
        rms_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
    hyper_compile = hyper_selection.compile(loss='mean_squared_error', metrics=[mae_metric, rms_metric])
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
    mae_valid = np.mean(val_mae[-5:])
    mae_5fold.append(mae_valid)
    all_test_index.append([train_index, test_index])

# Make output directories.
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
            label=r"Test: {0:0.4f} $\pm$ {1:0.4f} ".format(np.mean(mae_5fold), np.std(mae_5fold)) + data_unit, c='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Lipop training with '+ model_name)
plt.legend(loc='upper right', fontsize='medium')
plt.savefig(os.path.join(filepath, model_name + "_mae_lipop" + post_fix_file + ".png"))
plt.show()

# Plot predicted targets vs actual targets for last split.
true_test = scaler.inverse_transform(ytest)
pred_test = scaler.inverse_transform(model.predict(xtest))
plt.figure()
plt.scatter(pred_test, true_test, alpha=0.3, label="MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit + "]")
plt.plot(np.arange(np.amin(true_test), np.amax(true_test), 0.05),
         np.arange(np.amin(true_test), np.amax(true_test), 0.05), color='red')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Prediction Lipop with " + model_name)
plt.legend(loc='upper left', fontsize='x-large')
plt.savefig(os.path.join(filepath, model_name + "_predict_lipop" + post_fix_file + ".png"))
plt.show()

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + post_fix_file + ".npz"), all_test_index)

# Save hyper-parameter again, which were used for this fit.
save_json_file(hyper, os.path.join(filepath, model_name + "_hyper" + post_fix_file + ".json"))
