import numpy as np
import time
import os
import argparse

from kgcnn.utils.learning import LinearLearningRateScheduler
from sklearn.model_selection import KFold
from kgcnn.data.datasets.cora import CoraDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelection
from kgcnn.utils.plots import plot_train_test_loss

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Cora dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="GCN")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.",
                    default="hyper/hyper_cora.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification.
model_name = args["model"]
model_selection = ModelSelection()
make_model = model_selection.make_model(model_name)

# Hyper-parameter identification.
hyper_selection = HyperSelection(args["hyper"], model_name=model_name, dataset_name="Cora")
hyper = hyper_selection.get_hyper()

# Loading Cora Dataset
hyper_data = hyper['data']
dataset = CoraDataset().make_undirected_edges()
dataset_name = dataset.dataset_name
data_length = dataset.length
labels = dataset.node_labels

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
xtrain = dataloader.tensor(ragged=is_ragged)
ytrain = np.array(labels)

# Test Split
kf = KFold(**hyper_selection.k_fold())
split_indices = kf.split(X=np.arange(len(labels[0]))[:, None])

# Variables
history_list, test_indices_list = [], []
model, scaler, xtest, ytest = None, None, None, None

# Training on splits
for train_index, test_index in split_indices:
    # Make mode for current split.
    model = make_model(**hyper_selection.make_model())

    # Make training/validation mask to hide test labels from training.
    val_mask = np.zeros_like(labels[0][:, 0])
    train_mask = np.zeros_like(labels[0][:, 0])
    val_mask[test_index] = 1
    train_mask[train_index] = 1
    val_mask = np.expand_dims(val_mask, axis=0)  # One graph in batch
    train_mask = np.expand_dims(train_mask, axis=0)  # One graph in batch

    # Compile model with optimizer and loss.
    # Important to use weighted_metrics!
    model.compile(**hyper_selection.compile(loss='categorical_crossentropy', weighted_metrics=["categorical_accuracy"]))
    print(model.summary())

    # Training loop
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtrain, ytrain, val_mask),
                     sample_weight=train_mask,  # Hide validation data!
                     **hyper_selection.fit(epochs=100, validation_freq=10)
                     )
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directories.
filepath = hyper_selection.results_file_path()
postfix_file = hyper_selection.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name="categorical_accuracy", val_loss_name="val_categorical_accuracy",
                     model_name=model_name, data_unit="", dataset_name=dataset_name, filepath=filepath,
                     file_name="acc_cora" + postfix_file + ".png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

# Save hyper-parameter again, which were used for this fit.
hyper_selection.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))
