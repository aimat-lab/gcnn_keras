import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

from copy import deepcopy
from sklearn.model_selection import KFold
from kgcnn.data.datasets.PROTEINS import PROTEINSDatset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.utils.data import save_json_file, load_json_file
from kgcnn.hyper.datasets import DatasetHyperSelection


# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on PROTEINS dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="GIN")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.", default=None)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification
model_name = args["model"]
ms = ModelSelection()
make_model = ms.make_model(model_name)

# Hyper-parameter identification
if args["hyper"] is None:
    # Default hyper-parameter from DatasetHyperSelection if available
    hs = DatasetHyperSelection()
    hyper = hs.get_hyper("PROTEINS", model_name)
else:
    hyper = load_json_file(args["hyper"])

# Loading PROTEINS Dataset
hyper_data = hyper['data']
dataset = PROTEINSDatset()
data_name = dataset.dataset_name
data_length = dataset.length

# Data-set split
kf = KFold(n_splits=5, random_state=None, shuffle=True)
split_indices = kf.split(X=np.arange(data_length)[:, None])

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = dataset.graph_labels

# Set learning rate and epochs
hyper_train = hyper['training']
epo = hyper_train['fit']['epochs']
epostep = hyper_train['fit']['validation_freq']
batch_size = hyper_train['fit']['batch_size']

train_loss = []
test_loss = []
acc_5fold = []
model = None
for train_index, test_index in split_indices:

    # Make the model for current split.
    model = make_model(**hyper['model'])

    # Select train and test data.
    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = dataloader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = dataloader[test_index].tensor(ragged=is_ragged), labels[test_index]

    # Compile model with optimizer and loss
    optimizer = tf.keras.optimizers.get(deepcopy(hyper_train['optimizer']))
    cbks = [tf.keras.utils.deserialize_keras_object(x) for x in hyper_train['callbacks']]
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  weighted_metrics=['categorical_accuracy'])
    print(model.summary())

    # Start and time training
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtest, ytest),
                     callbacks=cbks,
                     **hyper_train['fit']
                     )
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history
    train_loss.append(np.array(hist.history['categorical_accuracy']))
    val_acc = np.array(hist.history['val_categorical_accuracy'])
    test_loss.append(val_acc)
    acc_valid = np.mean(val_acc[-10:])
    acc_5fold.append(acc_valid)

# Make output directories
os.makedirs(data_name, exist_ok=True)
filepath = os.path.join(data_name, hyper['model']['name'])
os.makedirs(filepath, exist_ok=True)

# Plot training- and test-loss vs epochs for all splits.
plt.figure()
for x in train_loss:
    plt.plot(np.arange(x.shape[0]), x, c='red', alpha=0.85)
for y in test_loss:
    plt.plot((np.arange(len(y)) + 1) * epostep, y, c='blue', alpha=0.85)
plt.scatter([train_loss[-1].shape[0]], [np.mean(acc_5fold)],
            label=r"Test: {0:0.4f} $\pm$ {1:0.4f}".format(np.mean(acc_5fold), np.std(acc_5fold)), c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('PROTEINS Loss')
plt.legend(loc='upper right', fontsize='large')
plt.savefig(os.path.join(filepath, 'acc_proteins.png'))
plt.show()

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
all_test_index = []
for train_index, test_index in split_indices:
    all_test_index.append([train_index, test_index])
np.savez(os.path.join(filepath, "kfold_splits.npz"), all_test_index)

# Save hyper-parameter again, which were used for this fit.
save_json_file(hyper, os.path.join(filepath, "hyper.json"))
