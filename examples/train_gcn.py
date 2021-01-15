import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from kgcnn.data.cora.cora import cora_graph
from kgcnn.literature.GCN import getmodelGCN,precompute_adjacency_scaled,scaled_adjacency_to_list
from kgcnn.utils.learning import lr_lin_reduction


# Download Dataset
A_data,X_data,y_data = cora_graph()
A_scaled = precompute_adjacency_scaled(A_data)
edge_index,edge_weight = scaled_adjacency_to_list(A_data,A_scaled)
nodes = X_data.todense()
edge_weight = np.expand_dims(edge_weight,axis=-1)
labels = np.expand_dims(y_data,axis=-1)
labels = np.array(labels==np.arange(70),dtype=np.float)

#Make ragged graph tensors
xtrain = [tf.RaggedTensor.from_row_lengths(nodes,np.array([len(nodes)],dtype=np.int)),
        tf.RaggedTensor.from_row_lengths(edge_index,np.array([len(edge_index)],dtype=np.int)),
        tf.RaggedTensor.from_row_lengths(edge_weight,np.array([len(edge_weight)],dtype=np.int)),
        ]

ytrain = np.expand_dims(labels,axis=0)


model = getmodelGCN(
            input_nodedim= 8710,
            input_type = "ragged",  #not used atm
            depth = 4,
            node_dim = 128,
            hidden_dim = 128,
            output_dim = [128,70],
            use_bias = True,
            activation = tf.keras.layers.LeakyReLU(alpha=0.1),
            graph_labeling = False,
            output_activ = 'softmax',
            )


learning_rate_start = 1e-3
learning_rate_stop = 1e-5
epo = 500
epomin = 400
optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)

cbks = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction(learning_rate_start,learning_rate_stop,epomin,epo))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics = ['accuracy'])

print(model.summary())

trainlossall = []
testlossall = []
validlossall = []

epostep = 10

start = time.process_time()

hist = model.fit(xtrain, ytrain, 
          epochs=epo,
          batch_size=1,
          callbacks=[cbks],
          verbose=1
          )

trainlossall = hist.history['accuracy']

stop = time.process_time()
print("Print Time for taining: ",stop - start)

trainlossall =np.array(trainlossall)


#Plot loss vs epochs    
plt.figure()
plt.plot(np.arange(trainlossall.shape[0]),trainlossall,label='Training Loss',c='blue')
#plt.plot(np.arange(epostep,epo+epostep,epostep),testlossall,label='Test Loss',c='red')
plt.xlabel('Epochs')
plt.ylabel('Accurarcy')
plt.title('GCN')
plt.legend(loc='lower right',fontsize='x-large')
plt.savefig('GCN_loss.png')
plt.show()