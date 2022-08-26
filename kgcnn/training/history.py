import numpy as np
import os
from kgcnn.data.utils import save_yaml_file


def save_history_score(
        histories: list,
        filepath: str = None,
        loss_name: str = None,
        val_loss_name: str = None,
        data_unit: str = "",
        model_name: str = "",
        file_name: str = "score.yaml",
        dataset_name: str = ""
):
    r"""Save fit results from fit histories to file.

    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        loss_name (str): Which loss or metric to pick from history. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
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

    result_dict = {}
    for i, l in zip(loss_name, train_loss):
        result_dict.update({i: [x[-1] for x in l]})
    for i, l in zip(val_loss_name, val_loss):
        result_dict.update({i: [x[-1] for x in l]})

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
                        val_loss_name[i], np.mean(y, axis=0)[-1], np.std(y, axis=0)[-1]) + data_unit,
                    color=vp[0].get_color()
                    )

    if filepath is not None:
        save_yaml_file(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))

    return fig
