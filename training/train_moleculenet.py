import logging
import numpy as np
import argparse
import os
import time

from datetime import timedelta
from kgcnn.data.moleculenet import MoleculeNetDataset
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
# From command line, one can specify the model, dataset and the hyper-parameters which contain all configuration
# for training and model setup.
parser = argparse.ArgumentParser(description='Train a GNN on a Molecule dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="DMPNN")
parser.add_argument("--dataset", required=False, help="Name of the dataset or leave empty for custom dataset.",
                    default="ESOLDataset")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_esol.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Get name for model, dataset, and path to a hyper-parameter file.
model_name = args["model"]
dataset_name = args["dataset"]
hyper_path = args["hyper"]

# A class `HyperSelection` is used to expose and verify hyper-parameters.
# The hyper-parameters are stores as a dictionary with section 'model', 'data' and 'training'.
hyper = HyperSelection(hyper_path, model_name=model_name, dataset_name=dataset_name)

# With `ModelSelection` a model definition from a module in kgcnn.literature can be loaded.
# At the moment there is a `make_model()` function in each module that sets up a keras model within the functional API
# of tensorflow-keras.
model_selection = ModelSelection(model_name)
make_model = model_selection.make_model()

# The `DatasetSelection` class is used to create a `MemoryGraphDataset` from config in hyper-parameters.
# The class also has functionality to check the dataset for properties or apply a series of methods on the dataset.
data_selection = DatasetSelection(dataset_name)

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# Those sub-classed classes are named after the dataset like e.g. `ESOLDataset`
try:
    dataset = data_selection.dataset(**hyper.dataset()["config"])

# If no name is given, a general `MoleculeNetDataset` is constructed.
# However, the construction then must be fully defined in the data section of the hyper-parameters,
# including all methods to run on the dataset. Information required in hyper-parameters are for example 'file_path',
# 'data_directory' etc. Making a custom training script rather than configuring the dataset via hyper-parameters can be
# more convenient.
except NotImplementedError:
    print("ERROR: Dataset not found, try general `MoleculeNetDataset`...")
    dataset = MoleculeNetDataset(**hyper.dataset()["config"])

# Set methods on the dataset to apply encoders or transformations or reload the data with different parameters.
# This is only done, if there is an entry with functional kwargs in hyper-parameters in the 'data' section.
# The `DatasetSelection` class first checks the `MoleculeNetDataset` and then tries each graph in the list to apply the
# methods listed by name below.
methods_supported = ["prepare_data", "read_in_memory", "set_attributes", "set_range", "set_angle",
                     "normalize_edge_weights_sym", "set_edge_indices_reverse"]
data_selection.perform_methods_on_dataset(dataset, methods_supported, hyper.data())

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
data_selection.assert_valid_model_input(dataset, hyper.inputs())

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper.inputs())
data_length = len(dataset)  # Length of the cleaned dataset.

# For `MoleculeNetDataset`, always train on `graph_labels`.
# Just making sure that the target is of shape `(N, #labels)`. This means output embedding is on graph level.
labels = np.array(dataset.obtain_property("graph_labels"))
if len(labels.shape) <= 1:
    labels = np.expand_dims(labels, axis=-1)

# Cross-validation via random KFold split form `sklearn.model_selection`. Other validation schemes could include
# stratified k-fold cross-validation for `MoleculeNetDataset` but is not implemented yet.
kf = KFold(**hyper.cross_validation()["config"])

# Iterate over the cross-validation splits.
# Indices for train-test splits are stored in 'test_indices_list'.
history_list, test_indices_list, model, hist, xtest, ytest, scaler = [], [], None, None, None, None, None
for train_index, test_index in kf.split(X=np.arange(data_length)[:, None]):

    # First select training and test graphs or molecules from indices, then convert them into tensorflow tensor
    # representation. Which property of the dataset and whether the tensor will be ragged is retrieved from the
    # kwargs of the keras `Input` layers ('name' and 'ragged').
    xtrain, ytrain = dataset[train_index].tensor(hyper.inputs()), labels[train_index]
    xtest, ytest = dataset[test_index].tensor(hyper.inputs()), labels[test_index]

    # Normalize training and test targets via a sklearn `StandardScaler`. No other scalers are used at the moment.
    # Scaler is applied to targets if 'scaler' appears in hyper-parameters. Only use for regression.
    if hyper.use_scaler():
        print("Using StandardScaler.")
        scaler = StandardScaler(**hyper.scaler()["config"])
        ytrain = scaler.fit_transform(ytrain)
        ytest = scaler.transform(ytest)

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

    # Make the model for current split using model kwargs from hyper-parameters.
    # The are always updated on top of the models default kwargs.
    model = make_model(**hyper.make_model())

    # Compile model with optimizer and loss from hyper-parameters. The metrics from this script is added to the hyper-
    # parameter entry for metrics.
    model.compile(**hyper.compile(metrics=metrics))
    print(model.summary())

    # Run keras model-fit and take time for training.
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtest, ytest),
                     **hyper.fit()
                     )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))

    # Get loss from history.
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

# Make output directory. This can further modified in hyper-parameters.
filepath = hyper.results_file_path()
postfix_file = hyper.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=model_name, data_unit=hyper.data_unit(), dataset_name=dataset_name,
                     filepath=filepath, file_name="loss" + postfix_file + ".png")

# Plot prediction for the last split.
predicted_y = model.predict(xtest)
true_y = ytest

# Predictions must be rescaled to original values.
if hyper.use_scaler():
    predicted_y = scaler.inverse_transform(predicted_y)
    true_y = scaler.inverse_transform(true_y)

# Plotting the prediction vs. true test targets for last split. Note for classification this is also done but
# can be ignored.
plot_predict_true(predicted_y, true_y,
                  filepath=filepath, data_unit=hyper.data_unit(),
                  model_name=model_name, dataset_name=dataset_name,
                  file_name="predict" + postfix_file + ".png")

# Save last keras-model to output-folder.
model.save(os.path.join(filepath, "model"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

# Save hyper-parameter again, which were used for this fit. Format is '.json'
# If non-serialized parameters were in the hyper-parameter config file, this operation may fail.
hyper.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))
