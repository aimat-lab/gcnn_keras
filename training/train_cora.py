import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

from kgcnn.utils.learning import LinearLearningRateScheduler
from sklearn.model_selection import KFold
from kgcnn.data.datasets.cora import CoraDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.datasets import DatasetHyperSelection
from kgcnn.utils.data import save_json_file, load_json_file

parser = argparse.ArgumentParser(description='Train a graph network on Cora dataset.')

parser.add_argument("--model", required=False, help="Graph model to train.", default="GCN")
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.", default=None)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model
model_name = args["model"]
ms = ModelSelection()
make_model = ms.make_model(model_name)

# Hyper
if args["hyper"] is None:
    # Default hyper-parameter
    hs = DatasetHyperSelection()
    hyper = hs.get_hyper("Cora", model_name)
else:
    hyper = load_json_file(args["hyper"])

# Loading PROTEINS Dataset
hyper_data = hyper['data']
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
hyper_train = hyper['training']
epo = hyper_train['fit']['epochs']
epostep = hyper_train['fit']['validation_freq']
# batch_size = hyper_train['fit']['batch_size']

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
    optimizer = tf.keras.optimizers.get(hyper_train['optimizer'])
    cbks = [tf.keras.utils.deserialize_keras_object(x) for x in hyper_train['callbacks']]
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

# Save hyper
save_json_file(hyper, os.path.join(filepath, "hyper.json"))