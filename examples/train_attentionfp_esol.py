import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

from tensorflow_addons.optimizers import AdamW

from kgcnn.literature.AttentiveFP import make_attentiveFP
from kgcnn.utils.data import ragged_tensor_from_nested_numpy
from kgcnn.utils.learning import lr_lin_reduction

from kgcnn.data.datasets.ESOL import ESOLDataset

dataset = ESOLDataset()
data_unit = "mol/L"
labels_val, nodes, edges, edge_indices, _ = dataset.get_graph()
scaler = StandardScaler(with_std=False, with_mean=True, copy=True)
labels = scaler.fit_transform(labels_val)

# Train Test split
labels_train, labels_test, nodes_train, nodes_test, edges_train, edges_test, edge_indices_train, edge_indices_test = train_test_split(
    labels, nodes, edges, edge_indices,  train_size=0.9, random_state=1)

# Convert to tf.RaggedTensor or tf.tensor
# adj_matrix copy of the data is generated by ragged_tensor_from_nested_numpy()
nodes_train, edges_train, edge_indices_train = ragged_tensor_from_nested_numpy(
    nodes_train), ragged_tensor_from_nested_numpy(edges_train), ragged_tensor_from_nested_numpy(
    edge_indices_train)

nodes_test, edges_test, edge_indices_test = ragged_tensor_from_nested_numpy(
    nodes_test), ragged_tensor_from_nested_numpy(edges_test), ragged_tensor_from_nested_numpy(
    edge_indices_test)

xtrain = nodes_train, edges_train, edge_indices_train
xtest = nodes_test, edges_test, edge_indices_test
ytrain = labels_train
ytest = labels_test

model = make_attentiveFP(
    input_node_shape=[None, 41],
    input_edge_shape=[None, 15],
    # Output
    output_embedd={"output_mode": 'graph', "output_type": 'padded'},
    output_mlp={"use_bias": [True, True], "units": [200, 1], "activation": ['kgcnn>leaky_relu', 'linear']},
    # model specs
    attention_args= {"units": 200, 'is_sorted': False, 'has_unconnected': True},
    depth=2,
    dropout=0.0
)

# Define learning rate and epochs
learning_rate_start = 10**-2.5
weight_decay = 10**-5
# learning_rate_stop = 1e-5
epo = 200
# epomin = 400
epostep = 5

# Compile model with optimizer and learning rate
# The scaled metric is meant to display the inverse-scaled mae values (optional)
# optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
optimizer = AdamW(lr=learning_rate_start, weight_decay=weight_decay)
# cbks = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction(learning_rate_start, learning_rate_stop, epomin, epo))
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()])
print(model.summary())


# Start training
start = time.process_time()
hist = model.fit(xtrain, ytrain,
                 epochs=epo,
                 batch_size=200,
                 # callbacks=[cbks],
                 validation_freq=epostep,
                 validation_data=(xtest, ytest),
                 verbose=2
                 )
stop = time.process_time()
print("Print Time for taining: ", stop - start)

trainlossall = np.array(hist.history['mean_absolute_error'])
testlossall = np.array(hist.history['val_mean_absolute_error'])

# Predict LUMO with model
pred_test = scaler.inverse_transform(model.predict(xtest))
true_test = scaler.inverse_transform(ytest)
mae_valid = np.mean(np.abs(pred_test - true_test))

# Plot loss vs epochs
plt.figure()
plt.plot(np.arange(trainlossall.shape[0]), trainlossall, label='Training Loss', c='blue')
plt.plot(np.arange(epostep, epo + epostep, epostep), testlossall, label='Test Loss', c='red')
plt.scatter([trainlossall.shape[0]], [mae_valid], label="{0:0.4f} ".format(mae_valid) + "[" + data_unit + "]", c='red')
plt.xlabel('Epochs')
plt.ylabel('Loss ' + "[" + data_unit + "]")
plt.title('Loss')
plt.legend(loc='upper right', fontsize='x-large')
plt.savefig('attentiveFP_loss.png')
plt.show()

# Predicted vs Actual
plt.figure()
plt.scatter(pred_test, true_test, alpha=0.3, label="MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit + "]")
plt.plot(np.arange(np.amin(true_test), np.amax(true_test), 0.05),
         np.arange(np.amin(true_test), np.amax(true_test), 0.05), color='red')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.legend(loc='upper left', fontsize='x-large')
plt.savefig('attentiveFP_predict.png')
plt.show()
