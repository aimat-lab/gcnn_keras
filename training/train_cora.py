import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

from copy import deepcopy
from kgcnn.utils.learning import LinearLearningRateScheduler
from sklearn.model_selection import KFold
from kgcnn.data.datasets.cora import CoraDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelectionTraining
from kgcnn.utils.data import save_json_file

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Cora dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="GCN")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.", default="hyper/hyper_cora.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification.
model_name = args["model"]
ms = ModelSelection()
make_model = ms.make_model(model_name)

# Hyper-parameter identification.
hyper_selection = HyperSelectionTraining(args["hyper"], model_name=model_name)
hyper = hyper_selection.get_hyper()

# Loading Cora Dataset
hyper_data = hyper['data']
dataset = CoraDataset().make_undirected_edges()
data_name = dataset.dataset_name
data_length = dataset.length
labels = dataset.node_labels

# Data-set split
k_fold_dict = hyper['training']["KFold"]
kf = KFold(**k_fold_dict)
split_indices = kf.split(X=np.arange(len(labels[0]))[:, None])

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
xtrain = dataloader.tensor(ragged=is_ragged)
ytrain = np.array(labels)

# Set epochs.
hyper_fit = hyper_selection.fit(epochs=100, validation_freq=10)
epo = hyper_fit['epochs']
epostep = hyper_fit['validation_freq']

train_loss = []
test_loss = []
acc_5fold = []
all_test_index = []
model = None
for train_index, test_index in split_indices:
    # Make mode for current split.
    model = make_model(**hyper['model'])

    # Make training/validation mask to hide test labels from training.
    val_mask = np.zeros_like(labels[0][:, 0])
    train_mask = np.zeros_like(labels[0][:, 0])
    val_mask[test_index] = 1
    train_mask[train_index] = 1
    val_mask = np.expand_dims(val_mask, axis=0)  # One graph in batch
    train_mask = np.expand_dims(train_mask, axis=0)  # One graph in batch

    # Compile model with optimizer and loss.
    hyper_compile = hyper_selection.compile(loss='categorical_crossentropy', weighted_metrics=["categorical_accuracy"])
    model.compile(**hyper_compile)
    print(model.summary())

    # Training loop
    trainloss_steps = []
    testloss_step = []
    start = time.process_time()
    hyper_fit_additional = hyper_selection.fit(epochs=100, validation_freq=10)
    hyper_fit_additional = {key: value for key, value in hyper_fit_additional.items() if key not in ["epochs",
        "batch_size", "initial_epoch", "sample_weight"]}

    for iepoch in range(0, epo, epostep):
        hist = model.fit(xtrain, ytrain,
                         epochs=iepoch + epostep,
                         initial_epoch=iepoch,
                         batch_size=1,
                         sample_weight=train_mask,  # Important!!!
                         **hyper_fit_additional
                         )

        trainloss_steps.append(hist.history)
        testloss_step.append(model.evaluate(xtrain, ytrain, sample_weight=val_mask))
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history.
    train_acc = np.concatenate([x['categorical_accuracy'] for x in trainloss_steps])
    train_loss.append(train_acc)
    val_acc = np.array(testloss_step)[:, 1]
    test_loss.append(val_acc)
    acc_valid = np.mean(val_acc[-5:])
    acc_5fold.append(acc_valid)
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
plt.scatter([train_loss[-1].shape[0]], [np.mean(acc_5fold)],
            label=r"Test: {0:0.4f} $\pm$ {1:0.4f}".format(np.mean(acc_5fold), np.std(acc_5fold)), c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Cora full 70 class loss for ' + model_name)
plt.legend(loc='upper right', fontsize='large')
plt.savefig(os.path.join(filepath, model_name + "_acc_cora" + post_fix_file + ".png"))
plt.show()

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + post_fix_file + ".npz"), all_test_index)

# Save hyper-parameter again, which were used for this fit.
save_json_file(hyper, os.path.join(filepath, model_name + "_hyper" + post_fix_file + ".json"))
