import matplotlib.pyplot as plt
import numpy as np
import os


def plot_train_test_loss(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = ""
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.

    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".

    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].history.keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].history.keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist.history[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist.history[x] for hist in histories])
        val_loss.append(loss)

    fig = plt.figure()
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    for i, y in enumerate(val_loss):
        val_step = len(train_loss[i][0]) / len(val_loss[i][0])
        vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                      label=val_loss_name[i])
        plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                         np.mean(y, axis=0) - np.std(y, axis=0),
                         np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
        plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                    label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                        val_loss_name[i], np.mean(y, axis=0)[-1], np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    plt.show()
    return fig


def plot_predict_true(y_predict, y_true, data_unit: list = None, model_name: str = "",
                      filepath: str = None, file_name: str = "", dataset_name: str = "", target_names: list = None
                      ):
    r"""Make a scatter plot of predicted versus actual targets. Not for k-splits.

    Args:
        y_predict (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        y_true (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        data_unit (list): String or list of string that matches `n_targets`. Name of the data's unit.
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        target_names (list): String or list of string that matches `n_targets`. Name of the targets.

    Returns:
        matplotlib.pyplot.figure: Figure of the scatter plot.
    """
    if len(y_predict.shape) == 1:
        y_predict = np.expand_dims(y_predict, axis=-1)
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=-1)
    num_targets = y_true.shape[1]

    if data_unit is None:
        data_unit = ""
    if isinstance(data_unit, str):
        data_unit = [data_unit]*num_targets
    if len(data_unit) != num_targets:
        print("WARNING:kgcnn: Targets do not match units for plot.")
    if target_names is None:
        target_names = ""
    if isinstance(target_names, str):
        target_names = [target_names]*num_targets
    if len(target_names) != num_targets:
        print("WARNING:kgcnn: Targets do not match names for plot.")

    fig = plt.figure()
    for i in range(num_targets):
        delta_valid = y_true[:, i] - y_predict[:, i]
        mae_valid = np.mean(np.abs(delta_valid[~np.isnan(delta_valid)]))
        plt.scatter(y_predict[:, i], y_true[:, i], alpha=0.3,
                    label=target_names[i] + " MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit[i] + "]")
    min_max = np.amin(y_true[~np.isnan(y_true)]), np.amax(y_true[~np.isnan(y_true)])
    plt.plot(np.arange(*min_max, 0.05), np.arange(*min_max, 0.05), color='red')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Prediction of " + model_name + " for " + dataset_name)
    plt.legend(loc='upper left', fontsize='x-large')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    plt.show()
    return fig
