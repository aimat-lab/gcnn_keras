import matplotlib.pyplot as plt
import numpy as np
import os


# Plot training- and test-loss vs epochs for all splits.
def plot_train_test_loss(histories: list, loss_name: str = "loss",
                         val_loss_name: str = "val_loss", data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = ""
                         ):
    # We assume multiple fits as in KFold.
    train_loss = []
    for hist in histories:
        train_mae = np.array(hist.history[loss_name])
        train_loss.append(train_mae)
    val_loss = []
    for hist in histories:
        val_mae = np.array(hist.history[val_loss_name])
        val_loss.append(val_mae)

    # Determine a mea
    mean_valid = [np.mean(x[-1:]) for x in val_loss]

    # val_step
    val_step = len(train_loss[0]) / len(val_loss[0])

    plt.figure()
    for x in train_loss:
        plt.plot(np.arange(x.shape[0]), x, c='red', alpha=0.85)
    for y in val_loss:
        plt.plot(np.arange(y.shape[0]) * val_step, y, c='blue', alpha=0.85)
    plt.scatter([train_loss[-1].shape[0]], [np.mean(mean_valid)],
                label=r"Test: {0:0.4f} $\pm$ {1:0.4f} ".format(np.mean(mean_valid), np.std(mean_valid)) + data_unit,
                c='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(loc='upper right', fontsize='medium')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + file_name))
    plt.show()


def plot_predict_true(y_predict, y_true, data_unit: str = "", model_name: str = "",
                      filepath: str = None, file_name: str = "", dataset_name: str = ""):
    mae_valid = np.mean(np.abs(y_true - y_predict))
    plt.figure()
    plt.scatter(y_predict, y_true, alpha=0.3, label="MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit + "]")
    plt.plot(np.arange(np.amin(y_true), np.amax(y_true), 0.05),
             np.arange(np.amin(y_true), np.amax(y_true), 0.05), color='red')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Prediction of " + model_name + " for " + dataset_name)
    plt.legend(loc='upper left', fontsize='x-large')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + file_name))
    plt.show()
