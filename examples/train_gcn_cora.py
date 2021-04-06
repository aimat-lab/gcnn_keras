import time

# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from kgcnn.data.cora.cora import cora_graph
from kgcnn.literature.GCN import make_gcn
from kgcnn.utils.adj import precompute_adjacency_scaled, scaled_adjacency_to_list, make_undirected
from kgcnn.utils.data import ragged_tensor_from_nested_numpy
from kgcnn.utils.learning import lr_lin_reduction

# Download and load Dataset
A_data, X_data, y_data = cora_graph()
# Make node features dense
nodes = X_data.todense()
# Precompute scaled and undirected (symmetric) adjacency matrix
A_scaled = precompute_adjacency_scaled(make_undirected(A_data))
# Use indices and weights instead of A
edge_index, edge_weight = scaled_adjacency_to_list(A_scaled)
edge_weight = np.expand_dims(edge_weight, axis=-1)
# Change labels to one-hot-encoding
labels = np.expand_dims(y_data, axis=-1)
labels = np.array(labels == np.arange(70), dtype=np.float)

# Make test/train split
# Since only one graph in the dataset
# Use a mask to hide test nodes labels
inds = np.arange(len(y_data))
ind_train, ind_val = train_test_split(inds, test_size=0.10, random_state=42)
val_mask = np.zeros_like(y_data)
train_mask = np.zeros_like(y_data)
val_mask[ind_val] = 1
train_mask[ind_train] = 1
val_mask = np.expand_dims(val_mask, axis=0)  # One graph in batch
train_mask = np.expand_dims(train_mask, axis=0)  # One graph in batch

# Make ragged graph tensors with 1 graph in batch
nodes, edges, edge_indices = ragged_tensor_from_nested_numpy([nodes]), ragged_tensor_from_nested_numpy(
    [edge_weight]), ragged_tensor_from_nested_numpy([edge_index])  # One graph in batch

# Set training data. But requires mask and batch-dimension of 1
xtrain = nodes, edges, edge_indices
ytrain = np.expand_dims(labels, axis=0)  # One graph in batch

model = make_gcn(
    input_node_shape=[None, 8710],
    input_edge_shape=[None, 1],
    # Output
    output_embedd={"output_mode": 'node'},
    output_mlp={"use_bias": [True, True, False], "units": [140, 70, 70], "activation": ['relu', 'relu', 'softmax']},
    # model specs
    depth=3,
    gcn_args={"units": 140, "use_bias": True, "activation": "relu", "has_unconnected": True}
)

# Set learning rate and epochs
learning_rate_start = 1e-3
learning_rate_stop = 1e-4
epo = 300
epomin = 260
epostep = 10

# Compile model with optimizer and loss
optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
cbks = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction(learning_rate_start, learning_rate_stop, epomin, epo))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              weighted_metrics=['categorical_accuracy'])
print(model.summary())

# Training loop
trainlossall = []
testlossall = []
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

    trainlossall.append(hist.history)
    testlossall.append(model.evaluate(xtrain, ytrain, sample_weight=val_mask))
stop = time.process_time()
print("Print Time for taining: ", stop - start)

# Pick out accuracy
testlossall = np.array(testlossall)
trainlossall = np.concatenate([x['categorical_accuracy'] for x in trainlossall])

# Plot loss vs epochs
plt.figure()
plt.plot(np.arange(1, len(trainlossall) + 1), trainlossall, label='Training Loss', c='blue')
plt.plot(np.arange(epostep, epo + epostep, epostep), testlossall[:, 1], label='Test Loss', c='red')
plt.xlabel('Epochs')
plt.ylabel('Accurarcy')
plt.title('GCN')
plt.legend(loc='lower right', fontsize='x-large')
plt.savefig('gcn_loss.png')
plt.show()
