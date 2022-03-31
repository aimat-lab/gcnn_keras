import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import time
import os
import kgcnn.utils.learning
import argparse

from datetime import timedelta
from tensorflow_addons import optimizers
from kgcnn.data.qm import QMDataset, QMGraphLabelScaler
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.selection.models import ModelSelection
from kgcnn.selection.hyper import HyperSelection
from kgcnn.selection.data import DatasetSelection
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true

# Input arguments from command line.
parser = argparse.ArgumentParser(description='Train a GNN on a QMDataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="Schnet")
parser.add_argument("--dataset", required=False, help="Name of the dataset or leave empty for custom dataset.",
                    default="QM9Dataset")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_qm9.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Main parameter about model, dataset, and hyper-parameter
model_name = args["model"]
dataset_name = args["dataset"]
hyper_path = args["hyper"]

# HyperSelection is used to store and verify hyperparameter.
hyper = HyperSelection(hyper_path, model_name=model_name, dataset_name=dataset_name)

# Model Selection to load a model definition from a module in kgcnn.literature
model_selection = ModelSelection(model_name)
make_model = model_selection.make_model()

# Loading a dataset from a module in kgcnn.data.datasets.
# If no name is given, a general QMDataset() is constructed.
# However, the construction then must be fully defined in the data section of the hyperparameter,
# including all methods to run on the dataset.
data_selection = DatasetSelection(dataset_name)

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# Those sub-classed classes are named after the dataset like e.g. `QM9Dataset`
# If no name is given, a general `QMDataset` is constructed.
# However, the construction then must be fully defined in the data section of the hyperparameter,
# including all methods to run on the dataset. Information required in hyperparameter are for example 'file_path',
# 'data_directory' etc. Making a custom training script rather than configuring the dataset via hyperparameter can be
# more convenient.
dataset = data_selection.dataset(**hyper.dataset())

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
data_selection.assert_valid_model_input(dataset, hyper.inputs())

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper.inputs())
data_length = len(dataset)  # Length of the cleaned dataset.

# For `QMDataset`, always train on graph, labels. The `QMDataset` has property 'label_names' and 'label_units', since
# targets can be in eV, Hartree, Bohr, GHz etc. Must be defined by subclasses of the dataset.
labels = np.array(dataset.obtain_property("graph_labels"))
label_names = dataset.obtain_property("label_names")
label_units = dataset.obtain_property("label_units")
if len(labels.shape) <= 1:
    labels = np.expand_dims(labels, axis=-1)

# Training on multiple targets for regression. This can is often required to train on orbital energies of ground
# or excited state or energies and enthalpies etc.
multi_target_indices = hyper.multi_target_indices()
if multi_target_indices is not None:
    labels = labels[:, multi_target_indices]
    if label_names is not None: label_names = [label_names[i] for i in multi_target_indices]
    if label_units is not None: label_units = [label_units[i] for i in multi_target_indices]
print("Labels %s in %s have shape %s" % (label_names, label_units, labels.shape))

# For QMDataset, also the atomic number is required to properly pre-scale extensive quantities like total energy.
atoms = dataset.node_number

# Cross-validation via random KFold split form `sklearn.model_selection`.
kf = KFold(**hyper.cross_validation()["config"])

# Training on splits. Since training on QM datasets can be expensive, there is a 'execute_splits' parameter to not
# train on all splits for testing.
execute_splits = hyper.execute_splits()
splits_done = 0
history_list, test_indices_list = [], []
model, hist, xtest, ytest, scaler, atoms_test = None, None, None, None, None, None
for train_index, test_index in kf.split(X=np.arange(data_length)[:, None]):

    # Only do execute_splits out of the k-folds of cross-validation.
    if splits_done >= execute_splits:
        break

    # Make the model for current split using model kwargs from hyperparameter.
    # The are always updated on top of the models default kwargs.
    model = make_model(**hyper.make_model())

    # First select training and test graphs from indices, then convert them into tensorflow tensor
    # representation. Which property of the dataset and whether the tensor will be ragged is retrieved from the
    # kwargs of the keras `Input` layers ('name' and 'ragged').
    xtrain, ytrain = dataset[train_index].tensor(hyper.inputs()), labels[train_index]
    xtest, ytest = dataset[test_index].tensor(hyper.inputs()), labels[test_index]
    # Also keep the same information for atomic numbers of the molecules.
    atoms_test = [atoms[i] for i in test_index]
    atoms_train = [atoms[i] for i in train_index]

    # Normalize training and test targets. For QM datasets this training script uses the `QMGraphLabelScaler` class.
    # Note that the QMGraphLabelScaler must receive a (serialized) list of individual scalers, one per each target.
    # These are extensive or intensive scalers, but could be expanded to have other normalization methods.
    if hyper.use_scaler():
        print("Using QMGraphLabelScaler.")
        scaler = QMGraphLabelScaler(**hyper.scaler()["config"]).fit(ytrain, atoms_train)  # Atomic number argument here!
        ytrain = scaler.fit_transform(ytrain, atoms_train)
        ytest = scaler.transform(ytest, atoms_test)

        # If scaler was used we add rescaled standard metrics to compile.
        scaler_scale = np.expand_dims(scaler.scale_, axis=0)
        mae_metric = ScaledMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
        rms_metric = ScaledRootMeanSquaredError(scaler_scale.shape, name="scaled_root_mean_squared_error")
        if scaler.scale_ is not None:
            mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
            rms_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
        metrics = [mae_metric, rms_metric]
    else:
        print("Not using QMGraphLabelScaler.")
        metrics = None

    # Compile model with optimizer and loss
    model.compile(**hyper.compile(loss="mean_absolute_error", metrics=metrics))
    print(model.summary())

    # Start and time training
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtest, ytest),
                     **hyper.fit())
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])
    splits_done = splits_done + 1

# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper.postfix_file()

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=model_name, data_unit=hyper.data_unit(), dataset_name=dataset_name,
                     filepath=filepath, file_name="loss" + postfix_file + ".png")

# Plot prediction
predicted_y = model.predict(xtest)
true_y = ytest

if hyper.use_scaler():
    predicted_y = scaler.inverse_transform(predicted_y, atoms_test)
    true_y = scaler.inverse_transform(true_y, atoms_test)

plot_predict_true(predicted_y, true_y,
                  filepath=filepath, data_unit=label_units,
                  model_name=model_name, dataset_name=dataset_name, target_names=label_names,
                  file_name="predict" + postfix_file + ".png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, "model" + postfix_file))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))
