import logging
import numpy as np
import argparse
import os
import time
import tensorflow as tf
# from tensorflow_addons import optimizers
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
from kgcnn.model.serial import deserialize as deserialize_model
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.training.hyper import HyperParameter
from kgcnn.utils.devices import set_devices_gpu

# Input arguments from command line with default values from example.
# From command line, one can specify the model, dataset and the hyperparameter which contain all configuration
# for training and model setup.
parser = argparse.ArgumentParser(description='Train a GNN on a Molecule dataset.')
parser.add_argument("--hyper", required=False, help="Filepath to hyperparameter config file (.py or .json).",
                    default="hyper/hyper_esol.py")
parser.add_argument("--category", required=False, help="Graph model to train.", default="GCN")
parser.add_argument("--model", required=False, help="Graph model to train.", default=None)
parser.add_argument("--dataset", required=False, help="Name of the dataset.", default=None)
parser.add_argument("--make", required=False, help="Name of the class for model.", default=None)
parser.add_argument("--gpu", required=False, help="GPU index used for training.", default=None, nargs="+", type=int)
parser.add_argument("--fold", required=False, help="Split or fold indices to run.", default=None, nargs="+", type=int)
parser.add_argument("--seed", required=False, help="Set random seed.", default=42, type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Set seed.
np.random.seed(args["seed"])
tf.random.set_seed(args["seed"])
tf.keras.utils.set_random_seed(args["seed"])

# Assigning GPU.
set_devices_gpu(args["gpu"])

# A class `HyperParameter` is used to expose and verify hyperparameter.
# The hyperparameter is stored as a dictionary with section 'model', 'dataset' and 'training'.
hyper = HyperParameter(
    hyper_info=args["hyper"], hyper_category=args["category"],
    model_name=args["model"], model_class=args["make"], dataset_class=args["dataset"])
hyper.verify()

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# Those sub-classed classes are named after the dataset like e.g. `ESOLDataset`
dataset = deserialize_dataset(hyper["dataset"])

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
time_list = []
history_list, model, hist, x_test, y_test, scaler = [], None, None, None, None, None
train_indices_all, test_indices_all = [], []
for train_index, test_index in kf.split(X=np.zeros((data_length, 1)), y=labels):
    test_indices_all.append(test_index)
    train_indices_all.append(train_index)

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
    model = deserialize_model(hyper["model"])

    # Compile model with optimizer and loss from hyperparameter.
    # The metrics from this script is added to the hyperparameter entry for metrics.
    model.compile(**hyper.compile(metrics=metrics))
    print(model.summary())

    # Run keras model-fit and take time for training.
    start = time.time()
    hist = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     **hyper.fit()
                     )
    stop = time.time()
    print("Print Time for training: %s" % str(timedelta(seconds=stop - start)))
    time_list.append(str(timedelta(seconds=stop - start)))

    # Get loss from history.
    history_list.append(hist)

# Make output directory. This can further be adapted in hyperparameter.
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Plot training- and test-loss vs epochs for all splits.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
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
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class,
                  file_name=f"predict{postfix_file}.png")

# Save last keras-model to output-folder.
model.save(os.path.join(filepath, f"model{postfix_file}"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{hyper.model_name}_test_indices_{postfix_file}.npz"), *test_indices_all)
np.savez(os.path.join(filepath, f"{hyper.model_name}_train_indices_{postfix_file}.npz"), *train_indices_all)

# Save hyperparameter again, which were used for this fit. Format is '.json'
# If non-serialized parameters were in the hyperparameter config file, this operation may fail.
hyper.save(os.path.join(filepath, f"{hyper.model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                   model_class=hyper.model_class,
                   model_version=model.__kgcnn_model_version__ if hasattr(model, "__kgcnn_model_version__") else "",
                   filepath=filepath, file_name=f"score{postfix_file}.yaml", time_list=time_list,
                   seed=args["seed"])
