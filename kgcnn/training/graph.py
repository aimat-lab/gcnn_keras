import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from tensorflow_addons import optimizers
from kgcnn.utils import learning
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.hyper.selection import HyperSelection
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from datetime import timedelta
from kgcnn.io.loader import NumpyTensorList


def train_graph_regression_supervised(X, y,
                                      make_model,
                                      hyper_selection,
                                      scaler,
                                      ):
    # Hyper-parameter via hyper_selection
    assert isinstance(hyper_selection, HyperSelection), "ERROR:kgcnn: Error require valid `HyperSelection`."
    hyper = hyper_selection.hyper()
    dataset_name = hyper_selection.dataset_name
    model_name = hyper_selection.model_name

    # Dataset-information
    data_length = len(y)

    # Test Split
    kf = KFold(**hyper_selection.k_fold())
    split_indices = kf.split(X=np.arange(data_length)[:, None])

    # Variables
    history_list, test_indices_list = [], []
    model, xtest, ytest = None, None, None

    # Training on splits
    for train_index, test_index in split_indices:
        # Select train and test data.
        is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
        xtrain, ytrain = X[train_index], y[train_index]
        xtest, ytest = X[test_index], y[test_index]

        if isinstance(xtrain, NumpyTensorList):
            xtrain = xtrain.tensor(ragged=is_ragged)
        if isinstance(xtest, NumpyTensorList):
            xtest = xtest.tensor(ragged=is_ragged)

        # Make model.
        model = make_model(**hyper_selection.make_model())

        # Normalize training and test targets.
        ytrain = scaler.fit_transform(ytrain)
        ytest = scaler.transform(ytest)

        # Get optimizer from serialized hyper-parameter.
        mae_metric = ScaledMeanAbsoluteError((1, 1), name='mean_absolute_error')
        rms_metric = ScaledRootMeanSquaredError((1, 1))
        if scaler.scale_ is not None:
            mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
            rms_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
        model.compile(**hyper_selection.compile(loss='mean_squared_error', metrics=[mae_metric, rms_metric]))
        print(model.summary())

        # Start and time training
        start = time.process_time()
        hist = model.fit(xtrain, ytrain,
                         validation_data=(xtest, ytest),
                         **hyper_selection.fit()
                         )
        stop = time.process_time()
        print("Print Time for training: ", str(timedelta(seconds=stop - start)))

        # Get loss from history
        history_list.append(hist)
        test_indices_list.append([train_index, test_index])

    # Make output directory
    filepath = hyper_selection.results_file_path()
    postfix_file = hyper_selection.postfix_file()

    # Plot training- and test-loss vs epochs for all splits.
    plot_train_test_loss(history_list, loss_name="mean_absolute_error", val_loss_name="val_mean_absolute_error",
                         model_name=model_name, data_unit="", dataset_name=dataset_name, filepath=filepath,
                         file_name="MAE" + postfix_file + ".png")
    # Plot prediction
    plot_predict_true(scaler.inverse_transform(model.predict(xtest)), scaler.inverse_transform(ytest),
                      filepath=filepath,
                      model_name=model_name, dataset_name=dataset_name,
                      file_name="predict" + postfix_file + ".png")

    # Save keras-model to output-folder.
    model.save(os.path.join(filepath, "model"))

    # Save original data indices of the splits.
    np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

    # Save hyper-parameter again, which were used for this fit.
    hyper_selection.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))


def train_graph_classification_supervised(X, y,
                                          make_model,
                                          hyper_selection,
                                          ):
    # Hyper-parameter via hyper_selection
    assert isinstance(hyper_selection, HyperSelection), "ERROR:kgcnn: Error require valid `HyperSelection`."
    hyper = hyper_selection.hyper()
    dataset_name = hyper_selection.dataset_name
    model_name = hyper_selection.model_name

    # Dataset-information
    data_length = len(y)
    labels = np.array(y)
    default_metric = "categorical_accuracy" if len(y.shape) > 1 and y.shape[-1] > 1 else "accuracy"
    default_loss = "categorical_crossentropy" if len(y.shape) > 1 and y.shape[-1] > 1 else "binary_crossentropy"

    # Test Split
    kf = KFold(**hyper_selection.k_fold())
    split_indices = kf.split(X=np.arange(data_length)[:, None])

    # Variables
    history_list, test_indices_list = [], []
    model, xtest, ytest = None, None, None

    # Train on splits
    for train_index, test_index in split_indices:
        # Select train and test data.
        is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
        xtrain, ytrain = X[train_index], labels[train_index]
        xtest, ytest = X[test_index], labels[test_index]

        if isinstance(xtrain, NumpyTensorList):
            xtrain = xtrain.tensor(ragged=is_ragged)
        if isinstance(xtest, NumpyTensorList):
            xtest = xtest.tensor(ragged=is_ragged)

        # Make the model for current split.
        model = make_model(**hyper_selection.make_model())

        # Compile model with optimizer and loss
        model.compile(**hyper_selection.compile(loss=default_loss, metrics=[default_metric]))
        print(model.summary())

        # Start and time training
        start = time.process_time()
        hist = model.fit(xtrain, ytrain,
                         validation_data=(xtest, ytest),
                         **hyper_selection.fit()
                         )
        stop = time.process_time()
        print("Print Time for training: ",  str(timedelta(seconds=stop - start)))

        # Get loss from history
        history_list.append(hist)
        test_indices_list.append([train_index, test_index])

    # Make output directories.
    filepath = hyper_selection.results_file_path()
    postfix_file = hyper_selection.postfix_file()

    # Plot training- and test-loss vs epochs for all splits.
    plot_train_test_loss(history_list, loss_name=default_metric, val_loss_name="val_"+default_metric,
                         model_name=model_name, data_unit="", dataset_name=dataset_name, filepath=filepath,
                         file_name="acc" + postfix_file + ".png")

    # Save keras-model to output-folder.
    model.save(os.path.join(filepath, "model"))

    # Save original data indices of the splits.
    np.savez(os.path.join(filepath, model_name + "_kfold_splits" + postfix_file + ".npz"), test_indices_list)

    # Save hyper-parameter again, which were used for this fit.
    hyper_selection.save(os.path.join(filepath, model_name + "_hyper" + postfix_file + ".json"))

