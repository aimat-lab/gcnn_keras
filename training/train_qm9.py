import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import kgcnn.utils.learning
import argparse

from tensorflow_addons import optimizers
from sklearn.preprocessing import StandardScaler
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from kgcnn.data.datasets.qm9 import QM9Dataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.utils.data import save_json_file, load_json_file
from kgcnn.hyper.datasets import DatasetHyperSelection

parser = argparse.ArgumentParser(description='Train a graph network on QM9 dataset.')

parser.add_argument("-m", "--model", required=False, help="Graph model to train.", default="Schnet")
parser.add_argument("-f", "--filepath", required=False, help="Filepath to hyper-parameter.", default=None)

args = vars(parser.parse_args())
print("Input of argpars:", args)

# Hyper and model
model_name = args["model"]
ms = ModelSelection()
make_model = ms.make_model(model_name)

# Info about data preparation
if args["filepath"] is None:
    # Default hyper-parameter
    hs = DatasetHyperSelection()
    hyper = hs.get_hyper("QM9")[model_name]
else:
    hyper = load_json_file(args["filepath"])

# Loading PROTEINS Dataset
hyper_data = hyper['data']
dataset = QM9Dataset().set_range(**hyper_data['range'])
data_name = dataset.dataset_name
data_unit = "eV"
data_length = dataset.length
target_names = dataset.target_names

data_points_to_use = hyper_data['data_points_to_use']  # Only for testing
data_selection = shuffle(np.arange(data_length))[:data_points_to_use]
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])[data_selection]
labels = dataset.graph_labels[data_selection][:, 6:9] * 27.2114  # Train on HOMO, LUMO, Eg
target_names = target_names[6:9]

# Data-set split
execute_splits = 2  # All splits may be too expensive for qm9
kf = KFold(n_splits=10, random_state=None, shuffle=True)
split_indices = kf.split(X=np.arange(len(labels))[:, None])

# Set learning rate and epochs
hyper_train = hyper['training']
epo = hyper_train['fit']['epochs']
epostep = hyper_train['fit']['validation_freq']
batch_size = hyper_train['fit']['batch_size']

train_loss = []
test_loss = []
mae_5fold = []
splits_done = 0
for train_index, test_index in split_indices:
    if splits_done >= execute_splits:
        break

    model = make_model(**hyper['model'])

    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = dataloader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = dataloader[test_index].tensor(ragged=is_ragged), labels[test_index]

    scaler = StandardScaler(with_std=True, with_mean=True, copy=True)
    ytrain = scaler.fit_transform(ytrain)
    ytest = scaler.transform(ytest)

    optimizer = tf.keras.optimizers.get(hyper_train['optimizer'])
    cbks = [tf.keras.utils.deserialize_keras_object(x) for x in hyper_train['callbacks']]
    mae_metric = ScaledMeanAbsoluteError((1, 3))
    rms_metric = ScaledRootMeanSquaredError((1, 3))
    if scaler.scale_ is not None:
        mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
        rms_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=[mae_metric, rms_metric])
    print(model.summary())

    # Start and time training
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     callbacks=cbks,
                     validation_data=(xtest, ytest),
                     **hyper_train['fit']
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

    splits_done += 1

# Make output directories
os.makedirs(data_name, exist_ok=True)
filepath = os.path.join(data_name, hyper['model']['name'])
os.makedirs(filepath, exist_ok=True)

# Plot loss vs epochs
plt.figure()
for x in train_loss:
    plt.plot(np.arange(x.shape[0]), x, c='red', alpha=0.85)
for y in test_loss:
    plt.plot((np.arange(len(y)) + 1) * epostep, y, c='blue', alpha=0.85)
plt.scatter([train_loss[-1].shape[0]], [np.mean(mae_5fold)],
            label=r"Test: {0:0.4f} $\pm$ {1:0.4f} ".format(np.mean(mae_5fold), np.std(mae_5fold)) + data_unit, c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('QM9 Loss')
plt.legend(loc='upper right', fontsize='medium')
plt.savefig(os.path.join(filepath, 'mae_qm9.png'))
plt.show()

# Predicted vs Actual
true_test = scaler.inverse_transform(ytest)
pred_test = scaler.inverse_transform(model.predict(xtest))
mae_last = np.mean(np.abs(true_test - pred_test), axis=0)
plt.figure()
for i in range(true_test.shape[-1]):
    plt.scatter(pred_test[:, i], true_test[:, i], alpha=0.3,
                label=target_names[i] + " MAE: {0:0.4f} ".format(mae_last[i]) + "[" + data_unit + "]")
plt.plot(np.arange(np.amin(true_test), np.amax(true_test), 0.05),
         np.arange(np.amin(true_test), np.amax(true_test), 0.05), color='red')
plt.xlabel('Predicted Last Split')
plt.ylabel('Actual')
plt.legend(loc='upper left', fontsize='x-small')
plt.savefig(os.path.join(filepath, 'predict_qm9.png'))
plt.show()

# Save hyper
save_json_file(hyper, os.path.join(filepath, "hyper.json"))

# Save model
model.save(os.path.join(filepath, "model"))

# save splits
all_test_index = []
for train_index, test_index in split_indices:
    all_test_index.append([data_selection[train_index], data_selection[test_index]])
np.savez(os.path.join(filepath, "kfold_splits.npz"), all_test_index)
