import logging
import numpy as np
import argparse
import os
import time
from tensorflow_addons import optimizers
from datetime import timedelta
from kgcnn.data.moleculenet import MoleculeNetDataset
import kgcnn.training.schedule
import kgcnn.training.scheduler
import kgcnn.metrics.loss
from kgcnn.training.history import save_history_score
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler as StandardLabelScaler
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.model.utils import get_model_class
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.hyper.hyper import HyperParameter
from kgcnn.utils.devices import set_devices_gpu

# Input arguments from command line with default values from example.
# From command line, one can specify the model, dataset and the hyperparameter which contain all configuration
# for training and model setup.
parser = argparse.ArgumentParser(description='Train a GNN on a Molecule dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="AttentiveFP")
parser.add_argument("--dataset", required=False, help="Name of the dataset or leave empty for custom dataset.",
                    default="ESOLDataset")
parser.add_argument("--hyper", required=False, help="Filepath to hyperparameter config file (.py or .json).",
                    default="hyper/hyper_esol.py")
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
# The hyperparameter stored as a dictionary with section 'model', 'data' and 'training'.
hyper = HyperParameter(hyper_path, model_name=model_name, model_class=make_function, dataset_name=dataset_name)

# With `ModelSelection` a model definition from a module in kgcnn.literature can be loaded.
# At the moment there is a `make_model()` function in each module that sets up a keras model within the functional API
# of tensorflow-keras.
make_model = get_model_class(model_name, make_function)

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# Those sub-classed classes are named after the dataset like e.g. `ESOLDataset`
dataset = deserialize_dataset(hyper["data"]["dataset"])

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper["model"]["config"]["inputs"])
data_length = len(dataset)  # Length of the cleaned dataset.

# For `MoleculeNetDataset`, always train on `graph_labels`.
# Just making sure that the target is of shape `(N, #labels)`. This means output embedding is on graph level.
labels = np.array(dataset.obtain_property("graph_labels"))
if len(labels.shape) <= 1:
    labels = np.expand_dims(labels, axis=-1)

# Cross-validation via random KFold split form `sklearn.model_selection`. Other validation schemes could include
# stratified k-fold cross-validation for `MoleculeNetDataset` but is not implemented yet.
kf = KFold(**hyper["training"]["cross_validation"]["config"])

# Iterate over the cross-validation splits.
# Indices for train-test splits are stored in 'test_indices_list'.
history_list, test_indices_list, model, hist, x_test, y_test, scaler = [], [], None, None, None, None, None
for train_index, test_index in kf.split(X=np.zeros((data_length, 1)), y=labels):

    # First select training and test graphs or molecules from indices, then convert them into tensorflow tensor
    # representation. Which property of the dataset and whether the tensor will be ragged is retrieved from the
    # kwargs of the keras `Input` layers ('name' and 'ragged').
    x_train, y_train = dataset[train_index].tensor(hyper["model"]["config"]["inputs"]), labels[train_index]
    x_test, y_test = dataset[test_index].tensor(hyper["model"]["config"]["inputs"]), labels[test_index]

    # Normalize training and test targets via a sklearn `StandardScaler`. No other scaler are used at the moment.
    # Scaler is applied to target if 'scaler' appears in hyperparameter. Only use for regression.
    if "scaler" in hyper["training"]:
        print("Using StandardScaler.")
        scaler = StandardLabelScaler(**hyper["training"]["scaler"]["config"])
        y_train = scaler.fit_transform(y_train)
        y_test = scaler.transform(y_test)

        # If scaler was used we add rescaled standard metrics to compile, since otherwise the keras history will not
        # directly log the original target values, but the scaled ones.
        mae_metric = ScaledMeanAbsoluteError((1, 1), name="scaled_mean_absolute_error")
        rms_metric = ScaledRootMeanSquaredError((1, 1), name="scaled_root_mean_squared_error")
        if scaler.scale_ is not None:
            mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
            rms_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
        metrics = [mae_metric, rms_metric]
    else:
        print("Not using StandardScaler.")
        metrics = None

    # Make the model for current split using model kwargs from hyperparameter.
    # They are always updated on top of the models default kwargs.
    model = make_model(**hyper["model"]["config"])

    # Compile model with optimizer and loss from hyperparameter.
    # The metrics from this script is added to the hyperparameter entry for metrics.
    model.compile(**hyper.compile(metrics=metrics))
    print(model.summary())

    # Run keras model-fit and take time for training.
    start = time.process_time()
    hist = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     **hyper.fit()
                     )
    stop = time.process_time()
    print("Print Time for training: %s" % str(timedelta(seconds=stop - start)))

    # Get loss from history.
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directory. This can further be adapted in hyperparameter.
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Plot training- and test-loss vs epochs for all splits.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=model_name, data_unit=data_unit, dataset_name=dataset_name,
                     filepath=filepath, file_name=f"loss{postfix_file}.png")

# Plot prediction for the last split.
predicted_y = model.predict(x_test)
true_y = y_test

# Predictions must be rescaled to original values.
if scaler:
    predicted_y = scaler.inverse_transform(predicted_y)
    true_y = scaler.inverse_transform(true_y)

# Plotting the prediction vs. true test targets for last split. Note for classification this is also done but
# can be ignored.
plot_predict_true(predicted_y, true_y,
                  filepath=filepath, data_unit=data_unit,
                  model_name=model_name, dataset_name=dataset_name,
                  file_name=f"predict{postfix_file}.png")

# Save last keras-model to output-folder.
model.save(os.path.join(filepath, f"model{postfix_file}"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{model_name}_kfold_splits{postfix_file}.npz"), test_indices_list)

# Save hyperparameter again, which were used for this fit. Format is '.json'
# If non-serialized parameters were in the hyperparameter config file, this operation may fail.
hyper.save(os.path.join(filepath, f"{model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=model_name, data_unit=data_unit, dataset_name=dataset_name,
                   model_class=make_function,
                   model_version=model.__kgcnn_model_version__ if hasattr(model, "__kgcnn_model_version__") else "",
                   filepath=filepath, file_name=f"score{postfix_file}.yaml")
