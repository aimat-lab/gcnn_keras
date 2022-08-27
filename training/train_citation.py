import numpy as np
import argparse
import os
import time
from tensorflow_addons import optimizers
from datetime import timedelta
import kgcnn.training.schedule
import kgcnn.training.scheduler
from kgcnn.training.history import save_history_score
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.hyper.hyper import HyperParameter
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.utils.models import get_model_class
from kgcnn.utils.devices import set_devices_gpu

# Input arguments from command line with default values from example.
# From command line, one can specify the model, dataset and the hyperparameter which contain all configuration
# for training and model setup.
parser = argparse.ArgumentParser(description='Train a GNN on a Citation dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="GATv2")
parser.add_argument("--dataset", required=False, help="Name of the dataset or leave empty for custom dataset.",
                    default="CoraLuDataset")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_cora_lu.py")
parser.add_argument("--make", required=False, help="Name of the make function or class for model.",
                    default="make_model")
parser.add_argument("--gpu", required=False, help="GPU index used for training.",
                    default=None, nargs="+", type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Get name for model, dataset, and path to a hyperparameter file.
model_name = args["model"]
dataset_name = args["dataset"]
hyper_path = args["hyper"]
make_function = args["make"]
gpu_to_use = args["gpu"]

# Assigning GPU.
set_devices_gpu(gpu_to_use)

# A class `HyperSelection` is used to expose and verify hyperparameter.
# The hyperparameter are stored as a dictionary with section 'model', 'data' and 'training'.
hyper = HyperParameter(hyper_path, model_name=model_name, model_class=make_function, dataset_name=dataset_name)

# With `ModelSelection` a model definition from a module in kgcnn.literature can be loaded.
# At the moment there is a `make_model()` function in each module that sets up a keras model within the functional API
# of tensorflow-keras.
make_model = get_model_class(model_name, make_function)

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# Those sub-classed classes are named after the dataset like e.g. `CoraLuDataset`
dataset = deserialize_dataset(hyper["data"]["dataset"])

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper["model"]["config"]["inputs"])
data_length = len(dataset)  # Length of the cleaned dataset.

# For Citation networks, node embedding tasks are assumed. Labels are taken as 'node_labels'.
# For now, node embedding tasks are restricted to a single graph, e.g. a citation network. Batch-dimension is one.
labels = dataset.obtain_property("node_labels")

# The complete graph is converted to a tensor here. Note that we still need a ragged tensor input, although it is not
# really needed for batch-dimension of one.
# Which property of the dataset and whether the tensor will be ragged is retrieved from the kwargs of the
# keras `Input` layers ('name' and 'ragged').
x_train = dataset.tensor(hyper["model"]["config"]["inputs"])
y_train = np.array(labels)

# Cross-validation via random KFold split form `sklearn.model_selection`.
kf = KFold(**hyper["training"]["cross_validation"]["config"])

# Iterate over the cross-validation splits.
# Indices for train-test splits are stored in 'test_indices_list'.
history_list, test_indices_list, model, hist = [], [], None, None
for train_index, test_index in kf.split(X=np.arange(len(labels[0]))[:, None]):

    # Make the model for current split using model kwargs from hyperparameter.
    # They are always updated on top of the models default kwargs.
    model = make_model(**hyper["model"]["config"])

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
    hist = model.fit(x_train, y_train,
                     validation_data=(x_train, y_train, val_mask),
                     sample_weight=train_mask,  # Hide validation data!
                     **hyper.fit(epochs=100, validation_freq=10)
                     )
    stop = time.process_time()
    print("Print Time for training: ", stop - start)

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directory. This can further be changed in hyperparameter.
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

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

# Save score of fit result for as text file.
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=model_name, data_unit="", dataset_name=dataset_name,
                   model_class=make_function,
                   filepath=filepath, file_name="score" + postfix_file + ".yaml")