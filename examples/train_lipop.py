import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from sklearn.preprocessing import StandardScaler
from tensorflow_addons.optimizers import AdamW
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.data.datasets.lipop import LipopDataset
from kgcnn.io.loader import NumpyTensorList

# Hyper
from kgcnn.literature.AttentiveFP import make_model

hyper = {'model': {'name': "AttentiveFP",
                   'inputs': [{'shape': (None, 41), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                              {'shape': (None, 15), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                              {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                   'input_embedding': {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                       "edge_attributes": {"input_dim": 5, "output_dim": 64}},
                   'output_embedding': 'graph',
                   'output_mlp': {"use_bias": [True, True], "units": [200, 1],
                                  "activation": ['kgcnn>leaky_relu', 'linear']},
                   'attention_args': {"units": 200},
                   'depth': 2,
                   'dropout': 0.2,
                   'verbose': 1
                   },
         'training': {'batch_size': 200, "learning_rate_start": 10**-2.5,
                      'epo': 200, 'epomin': 400, 'epostep': 10, "weight_decay": 10 ** -5
                      }
         }

# Loading PROTEINS Dataset
dataset = LipopDataset().define_attributes()
data_name = dataset.dataset_name
data_unit = "logD at pH 7.4"
data_length = dataset.data_length

# Data-set split
kf = KFold(n_splits=5, random_state=None, shuffle=True)
split_indices = kf.split(X=np.arange(data_length)[:, None])

dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = dataset.graph_labels

# Set learning rate and epochs
hyper_train = hyper['training']
learning_rate_start = hyper_train['learning_rate_start']
epo = hyper_train['epo']
epomin = hyper_train['epomin']
epostep = hyper_train['epostep']
batch_size = hyper_train['batch_size']
weight_decay = hyper_train['weight_decay']

train_loss = []
test_loss = []
mae_5fold = []
for train_index, test_index in split_indices:
    model = make_model(**hyper['model'])

    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = dataloader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = dataloader[test_index].tensor(ragged=is_ragged), labels[test_index]

    scaler = StandardScaler(with_std=True, with_mean=True, copy=True)
    ytrain = scaler.fit_transform(ytrain)
    ytest = scaler.transform(ytest)

    optimizer = AdamW(lr=learning_rate_start, weight_decay=weight_decay)
    mae_metric = ScaledMeanAbsoluteError((1, 1))
    rms_metric = ScaledRootMeanSquaredError((1, 1))
    if scaler.scale_ is not None:
        mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
        rms_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=[mae_metric, rms_metric])
    print(model.summary())

    # Start and time training
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     epochs=epo,
                     batch_size=batch_size,
                     callbacks=[],
                     validation_freq=epostep,
                     validation_data=(xtest, ytest),
                     verbose=2
                     )
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history
    train_mae = np.array(hist.history['mean_absolute_error'])
    train_loss.append(train_mae)
    val_mae = np.array(hist.history['val_mean_absolute_error'])
    test_loss.append(val_mae)
    mae_valid = np.mean(val_mae[-5:])
    mae_5fold.append(mae_valid)

os.makedirs(data_name, exist_ok=True)
filepath = os.path.join(data_name, hyper['model']['name'])
os.makedirs(filepath, exist_ok=True)

# Plot loss vs epochs
plt.figure()
for x in train_loss:
    plt.plot(np.arange(x.shape[0]), x, c='red', alpha=0.85)
for y in test_loss:
    plt.plot((np.arange(len(y)) + 1) * epostep, y, c='blue', alpha=0.85)
plt.scatter([train_loss[-1].shape[0]], [np.mean(mae_5fold)],
            label=r"Test: {0:0.4f} $\pm$ {1:0.4f} ".format(np.mean(mae_5fold), np.std(mae_5fold)) + data_unit, c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Lipop Loss')
plt.legend(loc='upper right', fontsize='medium')
plt.savefig(os.path.join(filepath, 'mae_lipop.png'))
plt.show()

# Predicted vs Actual
true_test = scaler.inverse_transform(ytest)
pred_test = scaler.inverse_transform(model.predict(xtest))
plt.figure()
plt.scatter(pred_test, true_test, alpha=0.3, label="MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit + "]")
plt.plot(np.arange(np.amin(true_test), np.amax(true_test), 0.05),
         np.arange(np.amin(true_test), np.amax(true_test), 0.05), color='red')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.legend(loc='upper left', fontsize='x-large')
plt.savefig(os.path.join(filepath, 'predict_lipop.png'))
plt.show()

# Save model
# model.save(os.path.join(filepath, "model"))

# save splits
all_test_index = []
for train_index, test_index in split_indices:
    all_test_index.append([train_index, test_index])
np.savez(os.path.join(filepath, "kfold_splits.npz"), all_test_index)
