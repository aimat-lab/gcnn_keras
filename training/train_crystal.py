import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
import time
import os
import argparse
from kgcnn.data.utils import save_pickle_file
from datetime import timedelta
from tensorflow_addons import optimizers
from kgcnn.data.transform.scaler.scaler import StandardLabelScaler
from kgcnn.data.transform.scaler.mol import QMGraphLabelScaler
import kgcnn.training.schedule
import kgcnn.training.scheduler
from kgcnn.training.history import save_history_score
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.hyper.hyper import HyperParameter
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.model.utils import get_model_class
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.utils.devices import set_devices_gpu

# Input arguments from command line.
parser = argparse.ArgumentParser(description='Train a GNN on a CrystalDataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="Schnet")
parser.add_argument("--dataset", required=False, help="Name of the dataset or leave empty for custom dataset.",
                    default="MatProjectPhononsDataset")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_mp_phonons.py")
parser.add_argument("--make", required=False, help="Name of the make function or class for model.",
                    default="make_crystal_model")
parser.add_argument("--gpu", required=False, help="GPU index used for training.",
                    default=None, nargs="+", type=int)
parser.add_argument("--fold", required=False, help="Split or fold indices to run.",
                    default=None, nargs="+", type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Main parameter about model, dataset, and hyperparameter
model_name = args["model"]
dataset_name = args["dataset"]
hyper_path = args["hyper"]
make_function = args["make"]
gpu_to_use = args["gpu"]
execute_folds = args["fold"]

# Assigning GPU.
set_devices_gpu(gpu_to_use)

# HyperParameter is used to store and verify hyperparameter.
hyper = HyperParameter(hyper_path, model_name=model_name, model_class=make_function, dataset_name=dataset_name)

# Model Selection to load a model definition from a module in kgcnn.literature
make_model = get_model_class(model_name, make_function)

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# Those sub-classed classes are named after the dataset like e.g. `MatProjectEFormDataset`
# If no name is given, a general `CrystalDataset` is constructed.
# However, the construction then must be fully defined in the data section of the hyperparameter,
# including all methods to run on the dataset. Information required in hyperparameter are for example 'file_path',
# 'data_directory' etc.
# Making a custom training script rather than configuring the dataset via hyperparameter can be
# more convenient.
dataset = deserialize_dataset(hyper["data"]["dataset"])

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper["model"]["config"]["inputs"])
data_length = len(dataset)  # Length of the cleaned dataset.

# Train on graph, labels. Must be defined by subclasses of the dataset.
labels = np.array(dataset.obtain_property("graph_labels"))
label_names = dataset.label_names
label_units = dataset.label_units
if len(labels.shape) <= 1:
    labels = np.expand_dims(labels, axis=-1)

# Training on multiple targets for regression.
multi_target_indices = hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper[
    "training"] else None
if multi_target_indices is not None:
    labels = labels[:, multi_target_indices]
    if label_names is not None:
        label_names = [label_names[i] for i in multi_target_indices]
    if label_units is not None:
        label_units = [label_units[i] for i in multi_target_indices]
print("Labels %s in %s have shape %s" % (label_names, label_units, labels.shape))

# For Crystals, also the atomic number is required to properly pre-scale extensive quantities like total energy.
atoms = dataset.obtain_property("node_number")

# Cross-validation via random KFold split form `sklearn.model_selection`.
kf = KFold(**hyper["training"]["cross_validation"]["config"])

# Training on splits. Since training on crystal datasets can be expensive, there is a 'execute_splits' parameter to not
# train on all splits for testing.
if "execute_folds" in hyper["training"]:
    execute_folds = hyper["training"]["execute_folds"]
splits_done = 0
history_list, test_indices_list = [], []
model, hist, x_test, y_test, scaler, atoms_test = None, None, None, None, None, None

for i, (train_index, test_index) in enumerate(kf.split(X=np.zeros((data_length, 1)), y=labels)):

    # Only do execute_splits out of the k-folds of cross-validation.
    if execute_folds:
        if i not in execute_folds:
            continue
    print("Running training on fold: %s" % i)

    # Make the model for current split using model kwargs from hyperparameter.
    # They are always updated on top of the models default kwargs.
    model = make_model(**hyper["model"]["config"])

    # First select training and test graphs from indices, then convert them into tensorflow tensor
    # representation. Which property of the dataset and whether the tensor will be ragged is retrieved from the
    # kwargs of the keras `Input` layers ('name' and 'ragged').
    x_train, y_train = dataset[train_index].tensor(hyper["model"]["config"]["inputs"]), labels[train_index]
    x_test, y_test = dataset[test_index].tensor(hyper["model"]["config"]["inputs"]), labels[test_index]
    # Also keep the same information for atomic numbers of the structures.
    atoms_test = [atoms[i] for i in test_index]
    atoms_train = [atoms[i] for i in train_index]

    # Normalize training and test targets via a sklearn `StandardScaler`. No other scaler are used at the moment.
    # Scaler is applied to target if 'scaler' appears in hyperparameter. Only use for regression.
    if "scaler" in hyper["training"]:
        print("Using StandardScaler.")
        if hyper["training"]["scaler"]["class_name"] == "QMGraphLabelScaler":
            scaler = QMGraphLabelScaler(**hyper["training"]["scaler"]["config"])
        else:
            scaler = StandardLabelScaler(**hyper["training"]["scaler"]["config"])

        y_train = scaler.fit_transform(y=y_train, atomic_number=atoms_train)
        y_test = scaler.transform(y=y_test, atomic_number=atoms_test)
        scaler_scale = scaler.get_scaling()

        # If scaler was used we add rescaled standard metrics to compile, since otherwise the keras history will not
        # directly log the original target values, but the scaled ones.
        mae_metric = ScaledMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
        rms_metric = ScaledRootMeanSquaredError(scaler_scale.shape, name="scaled_root_mean_squared_error")
        if scaler.scale_ is not None:
            mae_metric.set_scale(scaler_scale)
            rms_metric.set_scale(scaler_scale)
        metrics = [mae_metric, rms_metric]
    else:
        print("Not using StandardScaler.")
        metrics = None
    # Compile model with optimizer and loss
    model.compile(**hyper.compile(loss="mean_absolute_error", metrics=metrics))
    print(model.summary())

    # Start and time training
    start = time.process_time()
    hist = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     **hyper.fit())
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])
    splits_done = splits_done + 1

# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Plot training- and test-loss vs epochs for all splits.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=model_name, data_unit=data_unit, dataset_name=dataset_name,
                     filepath=filepath, file_name=f"loss{postfix_file}.png")

# Plot prediction
predicted_y = model.predict(x_test)
true_y = y_test

if scaler:
    predicted_y = scaler.inverse_transform(y=predicted_y, atomic_number=atoms_test)
    true_y = scaler.inverse_transform(y=true_y, atomic_number=atoms_test)

plot_predict_true(predicted_y, true_y,
                  filepath=filepath, data_unit=label_units,
                  model_name=model_name, dataset_name=dataset_name, target_names=label_names,
                  file_name=f"predict{postfix_file}.png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, f"model{postfix_file}"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{model_name}_kfold_splits{postfix_file}.npz"), test_indices_list)

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, f"{model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=model_name, data_unit=data_unit, dataset_name=dataset_name,
                   model_class=make_function, multi_target_indices=multi_target_indices, execute_folds=execute_folds,
                   filepath=filepath, file_name=f"score{postfix_file}.yaml")

# Save full history.
save_pickle_file([x.history for x in history_list],
                 os.path.join(filepath, f"histories_all{postfix_file}.pickle"))

# Save scaler.
scaler.save(os.path.join(filepath, f"scaler_{postfix_file}"))
