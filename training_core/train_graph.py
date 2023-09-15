import os
import keras_core as ks
import numpy as np
import argparse
import time
import kgcnn.training_core.scheduler
from datetime import timedelta
from kgcnn.training_core.history import save_history_score, load_history_list
from kgcnn.metrics_core.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.preprocessing import StandardScaler as StandardLabelScaler
from kgcnn.utils_core.plots import plot_train_test_loss, plot_predict_true
from kgcnn.model.serial import deserialize as deserialize_model
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.training_core.hyper import HyperParameter
from kgcnn.utils_core.devices import check_device
from kgcnn.data.utils import save_pickle_file

# Input arguments from command line with default values from example.
# From command line, one can specify the model, dataset and the hyperparameter which contain all configuration
# for training and model setup.
parser = argparse.ArgumentParser(description='Train a GNN on a graph regression or classification task.')
parser.add_argument("--hyper", required=False, help="Filepath to hyperparameter config file (.py or .json).",
                    default="hyper/hyper_esol.py")
parser.add_argument("--category", required=False, help="Graph model to train.", default="GAT")
parser.add_argument("--model", required=False, help="Graph model to train.", default=None)
parser.add_argument("--dataset", required=False, help="Name of the dataset.", default=None)
parser.add_argument("--make", required=False, help="Name of the class for model.", default=None)
parser.add_argument("--gpu", required=False, help="GPU index used for training.", default=None, nargs="+", type=int)
parser.add_argument("--fold", required=False, help="Split or fold indices to run.", default=None, nargs="+", type=int)
parser.add_argument("--seed", required=False, help="Set random seed.", default=42, type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Check for gpu
check_device()

# Set seed.
np.random.seed(args["seed"])
ks.utils.set_random_seed(args["seed"])

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

# Make output directory. This can further be adapted in hyperparameter.
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Always train on `graph_labels`.
# Just making sure that the target is of shape `(N, #labels)`. This means output embedding is on graph level.
labels, label_names, label_units = dataset.get_multi_target_indices(
    "graph_labels",
    hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper["training"] else None,
    data_unit=hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else None
)

# Iterate over the cross-validation splits.
# Indices for train-test splits are stored in 'test_indices_list'.
execute_folds = args["fold"] if "execute_folds" not in hyper["training"] else hyper["training"]["execute_folds"]
time_list, history_list = [], []
model, hist, x_test, y_test, scaler, current_split = None, None, None, None, None, None
train_indices_all, test_indices_all = [], []
for current_split, (train_index, test_index) in enumerate(kf.split(X=np.zeros((data_length, 1)), y=labels)):

    # Keep list of train/test indices.
    test_indices_all.append(test_index)
    train_indices_all.append(train_index)

    # Only do execute_splits out of the k-folds of cross-validation.
    if execute_folds:
        if current_split not in execute_folds:
            continue
    print("Running training on split: '%s'." % current_split)

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
    model.summary()
    print(" Compiled with jit: %s" % model._jit_compile)
    # Run keras model-fit and take time for training.
    start = time.time()
    hist = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     **hyper.fit()
                     )
    stop = time.time()
    print("Print Time for training: %s" % str(timedelta(seconds=stop - start)))
    time_list.append(str(timedelta(seconds=stop - start)))

    # Save history for this fold.
    save_pickle_file(hist.history, os.path.join(filepath, f"history{postfix_file}_fold_{current_split}.pickle"))

# Plot training- and test-loss vs epochs for all splits.
history_list = load_history_list(os.path.join(filepath, f"history{postfix_file}_fold_(i).pickle"), current_split+1)
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=hyper.model_name, data_unit=label_units, dataset_name=hyper.dataset_class,
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
plot_predict_true(predicted_y, true_y, target_names=label_names,
                  filepath=filepath, data_unit=label_units,
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class,
                  file_name=f"predict{postfix_file}.png")

# Save last keras-model to output-folder.
model.save(os.path.join(filepath, f"model{postfix_file}.keras"))

# Save last keras-model to output-folder.
model.save_weights(os.path.join(filepath, f"model{postfix_file}.weights.h5"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{hyper.model_name}_test_indices_{postfix_file}.npz"), *test_indices_all)
np.savez(os.path.join(filepath, f"{hyper.model_name}_train_indices_{postfix_file}.npz"), *train_indices_all)

# Save hyperparameter again, which were used for this fit. Format is '.json'
# If non-serialized parameters were in the hyperparameter config file, this operation may fail.
hyper.save(os.path.join(filepath, f"{hyper.model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=hyper.model_name, data_unit=label_units, dataset_name=hyper.dataset_class,
                   model_class=hyper.model_class,
                   model_version=model.__kgcnn_model_version__ if hasattr(model, "__kgcnn_model_version__") else "",
                   filepath=filepath, file_name=f"score{postfix_file}.yaml", time_list=time_list,
                   seed=args["seed"])
