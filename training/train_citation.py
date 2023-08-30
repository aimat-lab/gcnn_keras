import numpy as np
import argparse
import os
import time
import tensorflow as tf
from tensorflow_addons import optimizers
from datetime import timedelta
import kgcnn.training.schedule
import kgcnn.training.scheduler
from kgcnn.training.history import save_history_score
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.training.hyper import HyperParameter
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.model.serial import deserialize as deserialize_model
from kgcnn.utils.devices import set_devices_gpu

# Input arguments from command line.
# From command line, one must specify hyperparameter config file and optionally a category.
# Model and dataset information are optional.
parser = argparse.ArgumentParser(description='Train a GNN on a Citation dataset.')
parser.add_argument("--hyper", required=False, help="Filepath to hyperparameter config file (.py or .json).",
                    default="hyper/hyper_cora_lu.py")
parser.add_argument("--category", required=False, help="Graph model to train.", default="GraphSAGE")
parser.add_argument("--model", required=False, help="Graph model to train.", default=None)
parser.add_argument("--dataset", required=False, help="Name of the dataset.", default=None)
parser.add_argument("--make", required=False, help="Name of the class for model.", default=None)
parser.add_argument("--gpu", required=False, help="GPU index used for training.", default=None, nargs="+", type=int)
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
# Those sub-classed classes are named after the dataset like e.g. `CoraLuDataset`
dataset = deserialize_dataset(hyper["dataset"])

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
history_list, train_indices_all, test_indices_all, model, hist = [], [], [], None, None
for train_index, test_index in kf.split(X=np.arange(len(labels[0]))[:, None]):
    test_indices_all.append(test_index)
    train_indices_all.append(train_index)

    # Make the model for current split using model kwargs from hyperparameter.
    # They are always updated on top of the models default kwargs.
    model = deserialize_model(hyper["model"])

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
    start = time.time()
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_train, y_train, val_mask),
        sample_weight=train_mask,  # Hide validation data!
        **hyper.fit(epochs=100, validation_freq=10)
    )
    stop = time.time()
    print("Print Time for training: ", stop - start)

    # Get loss from history
    history_list.append(hist)

# Make output directory. This can further be changed in hyperparameter.
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Plot training- and test-loss vs epochs for all splits.
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=hyper.model_name, data_unit="", dataset_name=hyper.dataset_class, filepath=filepath,
                     file_name=f"loss{postfix_file}.png")

# Save keras-model to output-folder.
model.save(os.path.join(filepath, f"model{postfix_file}"))

# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{hyper.model_name}_test_indices_{postfix_file}.npz"), *test_indices_all)
np.savez(os.path.join(filepath, f"{hyper.model_name}_train_indices_{postfix_file}.npz"), *train_indices_all)

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, f"{hyper.model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                   model_class=hyper.model_class, seed=args["seed"],
                   filepath=filepath, file_name=f"score{postfix_file}.yaml")