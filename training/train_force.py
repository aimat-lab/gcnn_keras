import numpy as np
import time
import os
import argparse
import keras_core as ks
from datetime import timedelta
import kgcnn.training.schedule
import kgcnn.training.scheduler
from kgcnn.data.utils import save_pickle_file
from kgcnn.data.transform.scaler.serial import deserialize as deserialize_scaler
from kgcnn.utils.devices import check_device, set_gpu_device
from kgcnn.training.history import save_history_score, load_history_list, load_time_list
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.models.serial import deserialize as deserialize_model
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.training.hyper import HyperParameter
from kgcnn.losses.losses import ForceMeanAbsoluteError
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledForceMeanAbsoluteError
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler

# Input arguments from command line.
parser = argparse.ArgumentParser(description='Train a GNN on an Energy-Force Dataset.')
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_md17.py")
parser.add_argument("--category", required=False, help="Graph model to train.", default="Schnet.EnergyForceModel")
parser.add_argument("--model", required=False, help="Graph model to train.", default=None)
parser.add_argument("--dataset", required=False, help="Name of the dataset.", default=None)
parser.add_argument("--make", required=False, help="Name of the class for model.", default=None)
parser.add_argument("--module", required=False, help="Name of the module for model.", default=None)
parser.add_argument("--gpu", required=False, help="GPU index used for training.", default=None, nargs="+", type=int)
parser.add_argument("--fold", required=False, help="Split or fold indices to run.", default=None, nargs="+", type=int)
parser.add_argument("--seed", required=False, help="Set random seed.", default=42, type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Check and set device
if args["gpu"] is not None:
    set_gpu_device(args["gpu"])
check_device()

# Set seed.
np.random.seed(args["seed"])
ks.utils.set_random_seed(args["seed"])

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

# Always train on `energy` .
# Just making sure that the target is of shape `(N, #labels)`. This means output embedding is on graph level.
label_names, label_units = dataset.set_multi_target_labels(
    "energy",
    hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper["training"] else None,
    data_unit=hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else None
)

# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Training on splits. Since training on Force datasets can be expensive, there is a 'execute_splits' parameter to not
# train on all splits for testing. Can be set via command line or hyperparameter.
if "cross_validation" in hyper["training"]:
    from sklearn.model_selection import KFold
    splitter = KFold(**hyper["training"]["cross_validation"]["config"])
    train_test_indices = [
        (train_index, test_index) for train_index, test_index in splitter.split(X=np.zeros((data_length, 1)))]
else:
    train_test_indices = dataset.get_train_test_indices(train="train", test="test")
train_indices_all, test_indices_all = [], []

# Run Splits.
execute_folds = args["fold"] if "execute_folds" not in hyper["training"] else hyper["training"]["execute_folds"]
model, scaled_predictions, splits_done, current_split = None, False, 0, None
for current_split, (train_index, test_index) in enumerate(dataset.get_train_test_indices(train="train", test="test")):

    # Keep list of train/test indices.
    test_indices_all.append(test_index)
    train_indices_all.append(train_index)

    # Only do execute_splits out of the k-folds of cross-validation.
    if execute_folds:
        if current_split not in execute_folds:
            continue
    print("Running training on split: '%s'." % current_split)

    # Make the model for current split using model kwargs from hyperparameter.
    model = deserialize_model(hyper["model"])

    # First select training and test graphs from indices, then convert them into tensorflow tensor
    # representation. Which property of the dataset and whether the tensor will be ragged is retrieved from the
    dataset_train, dataset_test = dataset[train_index], dataset[test_index]

    # Normalize training and test targets.
    # For Force datasets this training script uses the `EnergyForceExtensiveLabelScaler` class.
    # Note that `EnergyForceExtensiveLabelScaler` uses both energy and forces for scaling.
    # Adapt output-scale via a transform.
    # Scaler is applied to target if 'scaler' appears in hyperparameter. Only use for regression.
    scaled_metrics = None
    if "scaler" in hyper["training"]:
        print("Using Scaler to adjust output scale of model.")
        scaler = deserialize_scaler(hyper["training"]["scaler"])
        scaler.fit_dataset(dataset_train)
        if hasattr(model, "set_scale"):
            print("Setting scale at model.")
            model.set_scale(scaler)
        else:
            print("Transforming dataset.")
            dataset_train = scaler.transform_dataset(dataset_train, copy_dataset=True, copy=True)
            dataset_test = scaler.transform_dataset(dataset_test, copy_dataset=True, copy=True)
            # If scaler was used we add rescaled standard metrics to compile, since otherwise the keras history will not
            # directly log the original target values, but the scaled ones.
            scaler_scale = scaler.get_scaling()
            mae_metric_energy = ScaledMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
            mae_metric_force = ScaledForceMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
            if scaler_scale is not None:
                mae_metric_energy.set_scale(scaler_scale)
                mae_metric_force.set_scale(scaler_scale)
            scaled_metrics = {"energy": [mae_metric_energy], "force": [mae_metric_force]}
            scaled_predictions = True

        # Save scaler to file
        scaler.save(os.path.join(filepath, f"scaler{postfix_file}_fold_{current_split}"))

    # Convert dataset to tensor information for model.
    x_train = dataset_train.tensor(hyper["model"]["config"]["inputs"])
    x_test = dataset_test.tensor(hyper["model"]["config"]["inputs"])

    # Compile model with optimizer and loss
    model.compile(**hyper.compile(
        loss={"energy": "mean_absolute_error", "force": ForceMeanAbsoluteError()},
        metrics=scaled_metrics))

    model.predict(x_test)
    # Model summary
    model.summary()
    print(" Compiled with jit: %s" % model._jit_compile)  # noqa

    # Convert targets into tensors.
    y_train = dataset_train.tensor(hyper["model"]["config"]["outputs"])
    y_test = dataset_test.tensor(hyper["model"]["config"]["outputs"])

    # Start and time training
    start = time.time()
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        **hyper.fit()
    )
    stop = time.time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))

    # Save history for this fold.
    save_pickle_file(hist.history, os.path.join(filepath, f"history{postfix_file}_fold_{current_split}.pickle"))
    save_pickle_file(str(timedelta(seconds=stop - start)),
                     os.path.join(filepath, f"time{postfix_file}_fold_{current_split}.pickle"))

    # Plot prediction
    predicted_y = model.predict(x_test, verbose=0)
    true_y = y_test

    plot_predict_true(np.array(predicted_y["energy"]), np.array(true_y["energy"]),
                      filepath=filepath, data_unit=label_units,
                      model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                      file_name=f"predict_energy{postfix_file}_fold_{splits_done}.png",
                      scaled_predictions=scaled_predictions)

    plot_predict_true(np.concatenate([np.array(f) for f in predicted_y["force"]], axis=0),
                      np.concatenate([np.array(f) for f in true_y["force"]], axis=0),
                      filepath=filepath, data_unit=label_units,
                      model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                      file_name=f"predict_force{postfix_file}_fold_{splits_done}.png",
                      scaled_predictions=scaled_predictions)

    # Save last keras-model to output-folder.
    model.save(os.path.join(filepath, f"model{postfix_file}_fold_{current_split}.keras"))

    # Save last keras-model to output-folder.
    model.save_weights(os.path.join(filepath, f"model{postfix_file}_fold_{current_split}.weights.h5"))

    # Get loss from history
    splits_done = splits_done + 1

# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{hyper.model_name}_test_indices_{postfix_file}.npz"), *test_indices_all)
np.savez(os.path.join(filepath, f"{hyper.model_name}_train_indices_{postfix_file}.npz"), *train_indices_all)

# Plot training- and test-loss vs epochs for all splits.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
history_list = load_history_list(os.path.join(filepath, f"history{postfix_file}_fold_(i).pickle"), current_split + 1)
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                     filepath=filepath, file_name=f"loss{postfix_file}.png")

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, f"{hyper.model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
time_list = load_time_list(os.path.join(filepath, f"time{postfix_file}_fold_(i).pickle"), current_split + 1)
save_history_score(
    history_list, loss_name=None, val_loss_name=None,
    model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
    model_class=hyper.model_class,
    multi_target_indices=hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper[
        "training"] else None,
    execute_folds=execute_folds, seed=args["seed"],
    filepath=filepath, file_name=f"score{postfix_file}.yaml",
    trajectory_name=(dataset.trajectory_name if hasattr(dataset, "trajectory_name") else None),
    time_list=time_list
)
