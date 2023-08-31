import numpy as np
import tensorflow as tf
import matplotlib as mpl
# mpl.use('Agg')
import time
import os
import argparse
from datetime import timedelta
from tensorflow_addons import optimizers, metrics
from kgcnn.data.qm import QMGraphLabelScaler
import kgcnn.training.schedule
import kgcnn.training.scheduler
from kgcnn.training.history import save_history_score
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.model.serial import deserialize as deserialize_model
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.training.hyper import HyperParameter
from kgcnn.utils.devices import set_devices_gpu

# Input arguments from command line.
parser = argparse.ArgumentParser(description='Train a GNN on a QMDataset.')
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_qm7.py")
parser.add_argument("--category", required=False, help="Graph model to train.", default="Schnet")
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

# HyperParameter is used to store and verify hyperparameter.
hyper = HyperParameter(
    hyper_info=args["hyper"], hyper_category=args["category"],
    model_name=args["model"], model_class=args["make"], dataset_class=args["dataset"])
hyper.verify()

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# Those sub-classed classes are named after the dataset like e.g. `QM9Dataset`
# If no name is given, a general `QMDataset` is constructed.
# However, the construction then must be fully defined in the data section of the hyperparameter,
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

# For `QMDataset`, always train on graph, labels. The `QMDataset` has property 'label_names' and 'label_units', since
# targets can be in eV, Hartree, Bohr, GHz etc. Must be defined by subclasses of the dataset.
labels = np.array(dataset.obtain_property("graph_labels"))
label_names = dataset.label_names
label_units = dataset.label_units
if len(labels.shape) <= 1:
    labels = np.expand_dims(labels, axis=-1)

# Training on multiple targets for regression. This can is often required to train on orbital energies of ground
# or excited state or energies and enthalpies etc.
multi_target_indices = hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper[
    "training"] else None
if multi_target_indices is not None:
    labels = labels[:, multi_target_indices]
    if label_names is not None:
        label_names = [label_names[i] for i in multi_target_indices]
    if label_units is not None:
        label_units = [label_units[i] for i in multi_target_indices]
print("Labels %s in %s have shape %s" % (label_names, label_units, labels.shape))

# For QMDataset, also the atomic number is required to properly pre-scale extensive quantities like total energy.
atoms = dataset.obtain_property("node_number")

# Cross-validation via random KFold split form `sklearn.model_selection`.
# Or from dataset information.
if hyper["training"]["cross_validation"] is None:
    print("Using dataset splits.")
    train_test_indices = dataset.get_split_indices()
else:
    kf = KFold(**hyper["training"]["cross_validation"]["config"])
    train_test_indices = [
        [train_index, test_index] for train_index, test_index in kf.split(X=np.zeros((data_length, 1)), y=labels)]

# Training on splits. Since training on QM datasets can be expensive, there is a 'execute_splits' parameter to not
# train on all splits for testing. Can be set via command line or hyperparameter.
execute_folds = args["fold"]
if "execute_folds" in hyper["training"]:
    execute_folds = hyper["training"]["execute_folds"]
splits_done = 0
history_list = []
model, hist, x_test, y_test, scaler, atoms_test = None, None, None, None, None, None
train_indices_all, test_indices_all = [], []
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
    # kwargs of the keras `Input` layers ('name' and 'ragged').
    x_train, y_train = dataset[train_index].tensor(hyper["model"]["config"]["inputs"]), labels[train_index]
    x_test, y_test = dataset[test_index].tensor(hyper["model"]["config"]["inputs"]), labels[test_index]
    # Also keep the same information for atomic numbers of the molecules.
    atoms_test = [atoms[i] for i in test_index]
    atoms_train = [atoms[i] for i in train_index]

    # Normalize training and test targets. For QM datasets this training script uses the `QMGraphLabelScaler` class.
    # Note that the QMGraphLabelScaler must receive a (serialized) list of individual scaler, one per each target.
    # These are extensive or intensive scaler, but could be expanded to have other normalization methods.
    if "scaler" in hyper["training"]:
        print("Using QMGraphLabelScaler.")
        # Atomic number argument here!
        scaler = QMGraphLabelScaler(**hyper["training"]["scaler"]["config"]).fit(y=y_train, atomic_number=atoms_train)
        y_train = scaler.transform(y=y_train, atomic_number=atoms_train)
        y_test = scaler.transform(y=y_test, atomic_number=atoms_test)

        # If scaler was used we add rescaled standard metrics to compile.
        scaler_scale = scaler.get_scaling()
        mae_metric = ScaledMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
        rms_metric = ScaledRootMeanSquaredError(scaler_scale.shape, name="scaled_root_mean_squared_error")
        if scaler.scale_ is not None:
            mae_metric.set_scale(scaler_scale)
            rms_metric.set_scale(scaler_scale)
        metrics = [mae_metric, rms_metric]
    else:
        print("Not using QMGraphLabelScaler.")
        metrics = None

    # Compile model with optimizer and loss
    model.compile(**hyper.compile(loss="mean_absolute_error", metrics=metrics))
    print(model.summary())

    # Start and time training
    start = time.time()
    hist = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     **hyper.fit())
    stop = time.time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))

    # Get loss from history
    history_list.append(hist)
    splits_done = splits_done + 1

# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Plot training- and test-loss vs epochs for all splits.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                     filepath=filepath, file_name=f"loss{postfix_file}.png")

# Plot prediction
predicted_y = model.predict(x_test, verbose=0)
true_y = y_test

if scaler:
    predicted_y = scaler.inverse_transform(y=predicted_y, atomic_number=atoms_test)
    true_y = scaler.inverse_transform(y=true_y, atomic_number=atoms_test)

plot_predict_true(predicted_y, true_y,
                  filepath=filepath, data_unit=label_units,
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                  file_name=f"predict{postfix_file}.png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, f"model{postfix_file}"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{hyper.model_name}_test_indices_{postfix_file}.npz"), *test_indices_all)
np.savez(os.path.join(filepath, f"{hyper.model_name}_train_indices_{postfix_file}.npz"), *train_indices_all)

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, f"{hyper.model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                   model_class=hyper.model_class, multi_target_indices=multi_target_indices,
                   execute_folds=execute_folds, seed=args["seed"],
                   filepath=filepath, file_name=f"score{postfix_file}.yaml")
