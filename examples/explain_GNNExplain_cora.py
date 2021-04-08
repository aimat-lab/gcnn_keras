import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from kgcnn.literature.GNNExplain import GNNExplainer,GNNExplainerOptimizer,GNNInterface

from kgcnn.utils.adj import precompute_adjacency_scaled,sort_edge_indices,make_adjacency_from_edge_indices,make_adjacency_undirected_logical_or,convert_scaled_adjacency_to_list
from kgcnn.utils.data import ragged_tensor_from_nested_numpy
from kgcnn.literature.GCN import make_gcn
from kgcnn.utils.learning import lr_lin_reduction

from kgcnn.data.cora.cora_lu import cora_graph

nodes, edge_index, labels, class_label_mapping = cora_graph()
nodes = nodes [:,1:] # Remove IDs
edge_index = sort_edge_indices(edge_index)
adj_matrix = make_adjacency_from_edge_indices(edge_index)
adj_matrix = precompute_adjacency_scaled(make_adjacency_undirected_logical_or(adj_matrix))
edge_index, edge_weight = convert_scaled_adjacency_to_list(adj_matrix)
edge_weight = np.expand_dims(edge_weight, axis=-1)
labels = np.expand_dims(labels, axis=-1)
labels = np.array(labels == np.arange(7), dtype=np.float)

# Make test/train split
# Since only one graph in the dataset
# Use a mask to hide test nodes labels
inds = np.arange(len(labels))
ind_train, ind_val = train_test_split(inds, test_size=0.10, random_state=42)
val_mask = np.zeros_like(labels)
train_mask = np.zeros_like(labels)
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
    input_node_shape=[None, 1432],
    input_edge_shape=[None, 1],
    # Output
    output_embedd={"output_mode": 'node'},
    output_mlp={"use_bias": [True, True, False], "units": [64, 16, 7], "activation": ['relu', 'relu', 'softmax']},
    # model specs
    depth=3,
    gcn_args={"units": 124, "use_bias": True, "activation": "relu", "has_unconnected": True}
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
