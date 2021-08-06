import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from kgcnn.utils.learning import LinearLearningRateScheduler
from sklearn.model_selection import KFold
from kgcnn.data.datasets.cora import CoraDataset
from kgcnn.io.loader import NumpyTensorList

# Hyper
from kgcnn.literature.GCN import make_model
hyper = {'model': {'name': "GCN",
                     'inputs': [{'shape': (None, 8710), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 1), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                                {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                     'input_embedding': {"node_attributes": {"input_dim": 95, "output_dim": 64},
                                         "edge_attributes": {"input_dim": 10, "output_dim": 64}},
                     'output_embedding': 'node',
                     'output_mlp': {"use_bias": [True, True, False], "units": [140, 70, 70],
                                    "activation": ['relu', 'relu', 'softmax']},
                     'gcn_args': {"units": 140, "use_bias": True, "activation": "relu"},
                     'depth': 3, 'verbose': 1
                   },
         'training': {'batch_size': 1, "learning_rate_start": 1e-3, 'learning_rate_stop': 1e-4,
                      'epo': 300, 'epomin': 260, 'epostep': 10
                      }
         }

# Loading PROTEINS Dataset
dataset = CoraDataset().make_undirected_edges()
data_name = dataset.dataset_name
data_length = dataset.length
labels = dataset.node_labels

# Data-set split
kf = KFold(n_splits=5, random_state=None, shuffle=True)
split_indices = kf.split(X=np.arange(len(labels[0]))[:, None])

dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
is_ragged = [x['ragged'] for x in hyper['model']['inputs']]
xtrain = dataloader.tensor(ragged=is_ragged)
ytrain = np.array(labels)


# Set learning rate and epochs
learning_rate_start = hyper['training']['learning_rate_start']
learning_rate_stop = hyper['training']['learning_rate_stop']
epo = hyper['training']['epo']
epomin = hyper['training']['epomin']
epostep = hyper['training']['epostep']
# batch_size = hyper['training']['batch_size']

train_loss = []
test_loss = []
acc_5fold = []
for train_index, test_index in split_indices:
    model = make_model(**hyper['model'])

    val_mask = np.zeros_like(labels[0][:,0])
    train_mask = np.zeros_like(labels[0][:,0])
    val_mask[test_index] = 1
    train_mask[train_index] = 1
    val_mask = np.expand_dims(val_mask, axis=0)  # One graph in batch
    train_mask = np.expand_dims(train_mask, axis=0)  # One graph in batch

    # Compile model with optimizer and loss
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
    cbks = LinearLearningRateScheduler(learning_rate_start, learning_rate_stop, epomin, epo)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  weighted_metrics=['categorical_accuracy'])
    print(model.summary())

    # Training loop
    trainloss_steps = []
    testloss_step = []
    start = time.process_time()
    for iepoch in range(0, epo, epostep):
        hist = model.fit(xtrain, ytrain,
                         epochs=iepoch + epostep,
                         initial_epoch=iepoch,
                         batch_size=1,
                         callbacks=[cbks],
                         verbose=1,
                         sample_weight=train_mask  # Important!!!
                         )

        trainloss_steps.append(hist.history)
        testloss_step.append(model.evaluate(xtrain, ytrain, sample_weight=val_mask))
    stop = time.process_time()
    print("Print Time for taining: ", stop - start)

    # Get loss from history
    train_acc = np.concatenate([x['categorical_accuracy'] for x in trainloss_steps])
    train_loss.append(train_acc)
    val_acc = np.array(testloss_step)[:,1]
    test_loss.append(val_acc)
    acc_valid = np.mean(val_acc[-5:])
    acc_5fold.append(acc_valid)

os.makedirs(data_name, exist_ok=True)
filepath = os.path.join(data_name, hyper['model']['name'])
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
plt.title('Cora Full 70 class Loss')
plt.legend(loc='upper right', fontsize='large')
plt.savefig(os.path.join(filepath, 'acc_cora.png'))
plt.show()

# Save model
model.save(os.path.join(filepath, "model"))

# save splits
all_test_index = []
for train_index, test_index in split_indices:
    all_test_index.append([train_index, test_index])
np.savez(os.path.join(filepath, "kfold_splits.npz"), all_test_index)
