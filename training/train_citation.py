import numpy as np
import argparse
import os
import time

from datetime import timedelta
from kgcnn.selection.models import ModelSelection
from kgcnn.selection.data import DatasetSelection
from kgcnn.selection.hyper import HyperSelection
from tensorflow_addons import optimizers
from kgcnn.utils import learning
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true

# Input arguments from command line with default values from example.
# From command line, one can specify the model, dataset and the hyperparameter which contain all configuration
# for training and model setup.
parser = argparse.ArgumentParser(description='Train a GNN on a Citation dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="GCN")
parser.add_argument("--dataset", required=False, help="Name of the dataset or leave empty for custom dataset.",
                    default="CoraDataset")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_cora.py")
parser.add_argument("--make", required=False, help="Name of the make function for model.",
                    default="make_model")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Get name for model, dataset, and path to a hyperparameter file.
model_name = args["model"]
dataset_name = args["dataset"]
hyper_path = args["hyper"]
make_function = args["make"]

# A class `HyperSelection` is used to expose and verify hyperparameter.
# The hyperparameter are stored as a dictionary with section 'model', 'data' and 'training'.
hyper = HyperSelection(hyper_path, model_name=model_name, dataset_name=dataset_name)

# With `ModelSelection` a model definition from a module in kgcnn.literature can be loaded.
# At the moment there is a `make_model()` function in each module that sets up a keras model within the functional API
# of tensorflow-keras.
model_selection = ModelSelection(model_name, make_function)
make_model = model_selection.make_model()

# The `DatasetSelection` class is used to create a `MemoryGraphDataset` from config in hyperparameter.
# The class also has functionality to check the dataset for properties or apply a series of methods on the dataset.
data_selection = DatasetSelection(dataset_name)

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# Those sub-classed classes are named after the dataset like e.g. `CoraLuDataset`
dataset = data_selection.dataset(**hyper.dataset())

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
data_selection.assert_valid_model_input(dataset, hyper.inputs())

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper.inputs())
data_length = len(dataset)

# For Citation networks, node embedding tasks are assumed. Labels are taken as 'node_labels'.
# For now, node embedding tasks are restricted to a single graph, e.g. a citation network. Batch-dimension is one.
labels = dataset.obtain_property("node_labels")

# The complete graph is converted to a tensor here. Note that we still need a ragged tensor input, although it is not
# really needed for batch-dimension of one.
# Which property of the dataset and whether the tensor will be ragged is retrieved from the kwargs of the
# keras `Input` layers ('name' and 'ragged').
xtrain = dataset.tensor(hyper.inputs())
ytrain = np.array(labels)

# Cross-validation via random KFold split form `sklearn.model_selection`.
kf = KFold(**hyper.cross_validation()["config"])

# Iterate over the cross-validation splits.
# Indices for train-test splits are stored in 'test_indices_list'.
history_list, test_indices_list, model, hist = [], [], None, None
for train_index, test_index in kf.split(X=np.arange(len(labels[0]))[:, None]):

    # Make the model for current split using model kwargs from hyperparameter.
    # They are always updated on top of the models default kwargs.
    model = model_selection(**hyper.make_model())

    # For semi-supervised learning with keras, revert to mask to hide nodes during training and for validation.
    val_mask = np.zeros_like(labels[0][:, 0])
    train_mask = np.zeros_like(labels[0][:, 0])
    val_mask[test_index] = 1
    train_mask[train_index] = 1
    # Requires one graph in the batch
    val_mask = np.expand_dims(val_mask, axis=0)
    train_mask = np.expand_dims(train_mask, axis=0)

    # Compile model with optimizer and loss from hyperparameter.
    # Since we use a sample weights for validation, the 'weighted_metrics' parameter has to be used for metrics.
    model.compile(**hyper.compile(weighted_metrics=None))
    print(model.summary())

    # Run keras model-fit and take time for training.
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtrain, ytrain, val_mask),
                     sample_weight=train_mask,  # Hide validation data!
                     **hyper.fit(epochs=100, validation_freq=10)
                     )
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directory. This can further be changed in hyperparameter.
filepath = hyper.results_file_path()
postfix_file = hyper.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=model_name, data_unit="", dataset_name=dataset_name, filepath=filepath,
                     file_name="loss" + postfix_file + ".png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))
