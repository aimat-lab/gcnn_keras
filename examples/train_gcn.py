import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from kgcnn.data.cora.cora import cora_graph
from kgcnn.literature.GCN import getmodelGCN
from kgcnn.utils.adj import precompute_adjacency_scaled,scaled_adjacency_to_list,make_undirected
from kgcnn.utils.learning import lr_lin_reduction


# Download Dataset
A_data,X_data,y_data = cora_graph()
A_scaled = precompute_adjacency_scaled(make_undirected(A_data))
edge_index,edge_weight = scaled_adjacency_to_list(A_scaled)
nodes = X_data.todense()
edge_weight = np.expand_dims(edge_weight,axis=-1)
labels = np.expand_dims(y_data,axis=-1)
labels = np.array(labels==np.arange(70),dtype=np.float)

#Make test/train split
inds = np.arange(len(y_data))
inds = shuffle(inds)
ind_val = inds[:1000]
ind_train = inds[1000:]
val_mask = np.zeros_like(y_data)
train_mask = np.zeros_like(y_data)
val_mask[ind_val] = 1
train_mask[ind_train] = 1
val_mask = np.expand_dims(val_mask,axis=0)
train_mask = np.expand_dims(train_mask,axis=0)

#Make ragged graph tensors
xtrain = [tf.RaggedTensor.from_row_lengths(nodes,np.array([len(nodes)],dtype=np.int)),
        tf.RaggedTensor.from_row_lengths(edge_weight,np.array([len(edge_weight)],dtype=np.int)),
        tf.RaggedTensor.from_row_lengths(edge_index,np.array([len(edge_index)],dtype=np.int)),
        ]

ytrain = np.expand_dims(labels,axis=0)


model = getmodelGCN(
                    input_node_shape = [None,8710],
                    input_edge_shape = [None,1],
                    input_state_shape = [1],
                    input_type = 'ragged', 
                    # Output
                    output_embedd = 'graph',
                    output_use_bias = [True,True,False],
                    output_dim = [140,70,70],
                    output_activation = ['relu',
                                         'relu',
                                         'softmax'],
                    output_type = 'padded',
                    #model specs
                    depth = 3,
                    node_dim = 140,
                    hidden_dim = 140,
                    use_bias = True,
                    activation = 'relu',
                    graph_labeling = False,
                    has_unconnected=True,
                    )


learning_rate_start = 1e-3
learning_rate_stop = 1e-4
epo = 300
epomin = 260
epostep = 10
optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)

cbks = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction(learning_rate_start,learning_rate_stop,epomin,epo))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              weighted_metrics = ['categorical_accuracy'])

print(model.summary())

trainlossall = []
testlossall = []
validlossall = []

start = time.process_time()

for iepoch in range(0,epo,epostep):

    hist = model.fit(xtrain, ytrain, 
              epochs=iepoch+epostep,
              initial_epoch=iepoch,
              batch_size=1,
              callbacks=[cbks],
              verbose=1,
              sample_weight = train_mask
              )

    trainlossall.append(hist.history)
    testlossall.append(model.evaluate(xtrain, ytrain,sample_weight=val_mask))    

stop = time.process_time()
print("Print Time for taining: ",stop - start)

testlossall = np.array(testlossall)
trainlossall = np.concatenate([x['categorical_accuracy'] for x in trainlossall])


#Plot loss vs epochs    
plt.figure()
plt.plot(np.arange(1,len(trainlossall)+1),trainlossall,label='Training Loss',c='blue')
plt.plot(np.arange(epostep,epo+epostep,epostep),testlossall[:,1],label='Test Loss',c='red')
plt.xlabel('Epochs')
plt.ylabel('Accurarcy')
plt.title('GCN')
plt.legend(loc='lower right',fontsize='x-large')
plt.savefig('GCN_loss.png')
plt.show()

