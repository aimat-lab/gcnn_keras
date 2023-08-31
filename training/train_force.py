import numpy as np
import tensorflow as tf
import matplotlib as mpl
# mpl.use('Agg')
import time
import os
import argparse
from datetime import timedelta
from tensorflow_addons import optimizers, metrics
import kgcnn.training.schedule
import kgcnn.training.scheduler
from kgcnn.training.history import save_history_score
from sklearn.model_selection import KFold
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.model.serial import deserialize as deserialize_model
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.training.hyper import HyperParameter
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, RaggedScaledMeanAbsoluteError
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.losses import RaggedMeanAbsoluteError
from kgcnn.data.utils import ragged_tensor_from_nested_numpy


# Input arguments from command line.
parser = argparse.ArgumentParser(description='Train a GNN on an Energy-Force Dataset.')
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_md17_revised.py")
parser.add_argument("--category", required=False, help="Graph model to train.", default="Schnet.EnergyForceModel")
parser.add_argument("--model", required=False, help="Graph model to train.", default=None)
parser.add_argument("--dataset", required=False, help="Name of the dataset.", default=None)
parser.add_argument("--make", required=False, help="Name of the class for model.", default=None)
parser.add_argument("--module", required=False, help="Name of the module for model.", default=None )
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

# HyperParameter is used to store and verify hyperparameter.
hyper = HyperParameter(
    hyper_info=args["hyper"], hyper_category=args["category"],
    model_name=args["model"], model_class=args["make"], dataset_class=args["dataset"], model_module=args["module"])
hyper.verify()

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# However, the construction must be fully defined in the data section of the hyperparameter,
# including all methods to run on the dataset. Information required in hyperparameter are for example 'file_path',
# 'data_directory' etc.
# Making a custom training script rather than configuring the dataset via hyperparameter can be
# more convenient.
dataset = deserialize_dataset(hyper["dataset"])

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper["model"]["config"]["inputs"])
data_length = len(dataset)  # Length of the cleaned dataset.

# Check that energy has at least one feature dimensions for e.g. multiple states.
energy = dataset.get("energy")
energy = np.array(energy)  # can handle energy as numpy array.
if len(energy.shape) <= 1:
    energy = np.expand_dims(energy, axis=-1)
dataset.set("energy", energy.tolist())

# Training on multiple targets for regression. This can is often required to train on orbital energies of ground
# or excited state or energies and enthalpies etc.
multi_target_indices = hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper[
    "training"] else None

# Provide label names and units.
label_names = dataset.label_names if hasattr(dataset, "label_names") else ""
label_units = dataset.label_units if hasattr(dataset, "label_units") else ""
print("Energy '%s' in '%s' has shape '%s'." % (label_names, label_units, energy.shape))

# Cross-validation via random KFold split form `sklearn.model_selection`.
# Or from dataset information.
if "cross_validation" in hyper["training"]:
    kf = KFold(**hyper["training"]["cross_validation"]["config"])
    train_test_indices = [
        [train_index, test_index] for train_index, test_index in kf.split(X=np.zeros((data_length, 1)), y=energy)]
elif "split_indices" in hyper["training"]:
    print("Using dataset splits.")
    train_test_indices = dataset.get_split_indices(**hyper["training"]["split_indices"])
elif "train_test_indices" in hyper["training"]:
    print("Using dataset train test indices.")
    train_test_indices = dataset.get_train_test_indices(**hyper["training"]["train_test_indices"])
else:
    raise ValueError("No information about model validation.")

# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Training on splits. Since training on Force datasets can be expensive, there is a 'execute_splits' parameter to not
# train on all splits for testing. Can be set via command line or hyperparameter.
execute_folds = args["fold"]
if "execute_folds" in hyper["training"]:
    execute_folds = hyper["training"]["execute_folds"]
splits_done = 0
history_list, test_indices_list = [], []
train_indices_all, test_indices_all = [], []
model, hist, x_test, scaler = None, None, None, None
for i, (train_index, test_index) in enumerate(train_test_indices):
    test_indices_all.append(test_index)
    train_indices_all.append(train_index)

    # Only do execute_splits out of the k-folds of cross-validation.
    if execute_folds:
        if i not in execute_folds:
            continue
    print("Running training on fold: %s" % i)

    # Make the model for current split using model kwargs from hyperparameter.
    # They are always updated on top of the models default kwargs.
    model = deserialize_model(hyper["model"])

    # First select training and test graphs from indices, then convert them into tensorflow tensor
    # representation. Which property of the dataset and whether the tensor will be ragged is retrieved from the
    dataset_train, dataset_test = dataset[train_index], dataset[test_index]

    # Normalize training and test targets.
    # For Force datasets this training script uses the `EnergyForceExtensiveLabelScaler` class.
    # Note that `EnergyForceExtensiveLabelScaler` uses both energy and forces for scaling.
    if "scaler" in hyper["training"]:
        print("Using `EnergyForceExtensiveLabelScaler`.")

        # Atomic number force and energy argument here!
        scaler_io = {"X": "atomic_number", "y": ["energy", "force"]}

        scaler = EnergyForceExtensiveLabelScaler(**hyper["training"]["scaler"]["config"])
        scaler.fit_dataset(dataset_train, **scaler_io)
        scaler.transform_dataset(dataset_train, **scaler_io)
        scaler.transform_dataset(dataset_test, **scaler_io)

        # If scaler was used we add rescaled standard metrics to compile.
        scaler_scale = scaler.get_scaling()
        mae_metric_energy = ScaledMeanAbsoluteError((1, 1), name="scaled_mean_absolute_error")
        mae_metric_force = RaggedScaledMeanAbsoluteError((1, 1), name="scaled_mean_absolute_error")
        if scaler_scale is not None:
            mae_metric_energy.set_scale(scaler_scale)
            mae_metric_force.set_scale(scaler_scale)
        metrics = {"energy": [mae_metric_energy], "force": [mae_metric_force]}
    else:
        print("Not using QMGraphLabelScaler.")
        scaler_io = None
        metrics = None

    # Convert dataset to tensor information for model.
    x_train = dataset_train.tensor(model["config"]["inputs"])
    x_test = dataset_test.tensor(model["config"]["inputs"])

    # Compile model with optimizer and loss
    model.compile(
        **hyper.compile(loss={"energy": "mean_absolute_error", "force": RaggedMeanAbsoluteError()}, metrics=metrics))
    model.predict(x_test)
    print(model.summary())

    # Convert targets into tensors.
    labels_in_dataset = {
        "energy": {"name": "energy", "ragged": False},
        "force": {"name": "force", "shape": (None, 3), "ragged": True}
    }
    y_train = dataset_train.tensor(labels_in_dataset)
    y_test = dataset_test.tensor(labels_in_dataset)

    # Start and time training
    start = time.time()
    hist = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     **hyper.fit())
    stop = time.time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])
    splits_done = splits_done + 1

    # Plot prediction
    predicted_y = model.predict(x_test, verbose=0)
    true_y = y_test

    if scaler:
        scaler.inverse_transform_dataset(dataset_train, **scaler_io)
        scaler.inverse_transform_dataset(dataset_test, **scaler_io)
        true_y = dataset_test.tensor(labels_in_dataset)

        predicted_y = scaler.inverse_transform(
            y=(predicted_y["energy"], predicted_y["force"]), X=dataset_test.get("atomic_number"))

    plot_predict_true(np.array(predicted_y[0]), np.array(true_y["energy"]),
                      filepath=filepath, data_unit=label_units,
                      model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                      file_name=f"predict_energy{postfix_file}_fold_{splits_done}.png")

    plot_predict_true(np.concatenate([np.array(f) for f in predicted_y[1]], axis=0),
                      np.concatenate([np.array(f) for f in true_y["force"]], axis=0),
                      filepath=filepath, data_unit=label_units,
                      model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                      file_name=f"predict_force{postfix_file}_fold_{splits_done}.png")

    # Save keras-model to output-folder.
    model.save(os.path.join(filepath, f"model{postfix_file}_fold_{splits_done}"))


# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{hyper.model_name}_test_indices_{postfix_file}.npz"), *test_indices_all)
np.savez(os.path.join(filepath, f"{hyper.model_name}_train_indices_{postfix_file}.npz"), *train_indices_all)

# Plot training- and test-loss vs epochs for all splits.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                     filepath=filepath, file_name=f"loss{postfix_file}.png")

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, f"{hyper.model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                   model_class=hyper.model_class,
                   multi_target_indices=multi_target_indices,
                   execute_folds=execute_folds, seed=args["seed"],
                   filepath=filepath, file_name=f"score{postfix_file}.yaml",
                   trajectory_name=(dataset.trajectory_name if hasattr(dataset, "trajectory_name") else None))
