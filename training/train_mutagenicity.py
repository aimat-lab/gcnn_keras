import numpy as np
import time
import os
import argparse

from kgcnn.utils.learning import LinearLearningRateScheduler
from sklearn.model_selection import KFold
from kgcnn.data.datasets.mutagenicity import MutagenicityDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelectionTraining
from kgcnn.utils.plots import plot_train_test_loss

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Mutagenicity dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="GraphSAGE")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.",
                    default="hyper/hyper_mutagenicity.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification.
model_name = args["model"]
model_selection = ModelSelection()
make_model = model_selection.make_model(model_name)

# Hyper-parameter identification.
hyper_selection = HyperSelectionTraining(args["hyper"], model_name=model_name, dataset_name="Mutagenicity")
hyper = hyper_selection.get_hyper()

# Loading Mutagenicity Dataset
hyper_data = hyper['data']
dataset = MutagenicityDataset()
dataset_name = dataset.dataset_name
data_length = dataset.length
if "set_edge_indices_reverse" in hyper_data:
    dataset.set_edge_indices_reverse()

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = np.array(dataset.graph_labels)

# Test Split
kf = KFold(**hyper_selection.k_fold())
split_indices = kf.split(X=np.arange(data_length)[:, None])

# Variables
history_list, test_indices_list = [], []
model, scaler, xtest, ytest = None, None, None, None

# Train on splits
for train_index, test_index in split_indices:
    # Select train and test data.
    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = dataloader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = dataloader[test_index].tensor(ragged=is_ragged), labels[test_index]

    # Make model.
    model = make_model(**hyper_selection.make_model())

    # Compile model with optimizer and loss
    model.compile(**hyper_selection.compile(loss='binary_crossentropy', metrics=['accuracy']))
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

# Make output directories.
filepath = hyper_selection.results_file_path()
postfix_file = hyper_selection.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name="accuracy", val_loss_name="val_accuracy",
                     model_name=model_name, data_unit="", dataset_name=dataset_name, filepath=filepath,
                     file_name="acc_mutagenicity" + postfix_file + ".png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

# Save hyper-parameter again, which were used for this fit.
hyper_selection.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))
