import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from kgcnn.utils.learning import LinearLearningRateScheduler
from sklearn.model_selection import KFold
from kgcnn.data.datasets.mutagenicity import MutagenicityDataset
from kgcnn.io.loader import NumpyTensorList

# Hyper
from kgcnn.literature.GIN import make_model

hyper = {'model': {'name': "GraphSAGE",
                   'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                              {'shape': (None,), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                              {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                   'input_embedding': {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                       "edge_attributes": {"input_dim": 5, "output_dim": 64}},
                   'output_embedding': 'graph',
                   'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                  "activation": ['relu', 'relu', 'sigmoid']},
                   'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                   'edge_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                   'pooling_args': {'pooling_method': "segment_mean"}, 'gather_args': {}, 'concat_args': {"axis": -1},
                   'use_edge_features': True, 'pooling_nodes_args': {'pooling_method': "mean"},
                   'depth': 3, 'verbose': 1
                   },
         'training': {'batch_size': 32, "learning_rate": 1e-2, 'epo': 150
                      }
         }

# Loading PROTEINS Dataset
dataset = MutagenicityDataset()
data_name = dataset.dataset_name
data_length = dataset.data_length

# Data-set split
kf = KFold(n_splits=5, random_state=None, shuffle=True)
split_indices = kf.split(X=np.arange(data_length)[:, None])

dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = dataset.graph_labels

# Set learning rate and epochs
learning_rate_start = 1e-3
learning_rate_stop = 1e-4
epo = 250
epomin = 260
epostep = 10
batch_size = 32

train_loss = []
test_loss = []
acc_5fold = []
for train_index, test_index in split_indices:
    model = make_model(**hyper['model'])

    is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
    xtrain, ytrain = dataloader[train_index].tensor(ragged=is_ragged), labels[train_index]
    xtest, ytest = dataloader[test_index].tensor(ragged=is_ragged), labels[test_index]

    # Compile model with optimizer and loss
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
    cbks = LinearLearningRateScheduler(learning_rate_start, learning_rate_stop, epomin, epo)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  weighted_metrics=['accuracy'])
    print(model.summary())

    # Start and time training
    start = time.process_time()
    hist = model.fit(xtrain, ytrain,
                     epochs=epo,
                     batch_size=batch_size,
                     callbacks=[cbks],
                     validation_freq=epostep,
                     validation_data=(xtest, ytest),
                     verbose=2
                     )
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history
    train_loss.append(np.array(hist.history['categorical_accuracy']))
    val_acc = np.array(hist.history['val_categorical_accuracy'])
    test_loss.append(val_acc)
    acc_valid = np.mean(val_acc[-10:])
    acc_5fold.append(acc_valid)

os.makedirs("mutagenicity", exist_ok=True)
filepath = os.path.join("mutagenicity", hyper['model']['name'])
os.makedirs(filepath, exist_ok=True)

# Plot loss vs epochs
plt.figure()
for x in train_loss:
    plt.plot(np.arange(x.shape[0]), x, c='red', alpha=0.85)
for y in test_loss:
    plt.plot((np.arange(len(y)) + 1) * epostep, y, c='blue', alpha=0.85)
plt.scatter([train_loss[-1].shape[0]], [np.mean(acc_5fold)],
            label=r"Test: {0:0.4f} $\pm$ {1:0.4f}".format(np.mean(acc_5fold), np.std(acc_5fold)), c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('mutagenicity Loss')
plt.legend(loc='upper right', fontsize='large')
plt.savefig(os.path.join(filepath, 'gin_proteins.png'))
plt.show()

# Save model
model.save(os.path.join(filepath, "model"))

# save splits
all_test_index = []
for train_index, test_index in split_indices:
    all_test_index.append([train_index, test_index])
np.savez(os.path.join(filepath, "kfold_splits.npz"), all_test_index)
