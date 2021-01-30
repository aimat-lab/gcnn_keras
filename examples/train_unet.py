import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from kgcnn.data.mutagen.mutag import mutag_graph
from kgcnn.literature.Unet import getmodelUnet
from kgcnn.utils.learning import lr_lin_reduction
from kgcnn.utils.adj import add_self_loops_to_indexlist

# Download Dataset
mutag_data = mutag_graph()
y_data = mutag_data[0]
y_data[y_data<0] = 0
y_data = np.expand_dims(y_data,axis=-1)
x_data = [mutag_data[1],None,mutag_data[2]]
x_data[2] = [add_self_loops_to_indexlist(x) for x in x_data[2]]
x_data[1] = [np.ones_like(x,dtype=np.float) for x in x_data[2]]
x_data.append(np.array([[len(x)] for x in mutag_data[1]]))

#Make test/train split
inds = np.arange(len(y_data))
inds = shuffle(inds)
ind_val = inds[:40]
ind_train = inds[40:]

# Select train/test data
xtrain = [[x[i] for i in ind_train] for x in x_data]
ytrain = y_data[ind_train]
xval = [[x[i] for i in ind_val] for x in x_data]
yval = y_data[ind_val]

def make_ragged(inlist):
    return tf.RaggedTensor.from_row_lengths(np.concatenate(inlist,axis=0), np.array([len(x) for x in inlist],dtype=np.int))

#Make ragged graph tensors plus normal tensor for graph state
xval = [make_ragged(x) for x in xval[:3]]   + [tf.constant(xval[3])] 
xtrain = [make_ragged(x) for x in xtrain[:3]]  + [tf.constant(xtrain[3])] 


model = getmodelUnet(
                input_node_shape = [None],
                input_edge_shape = [None,1],
                input_state_shape = [1],
                input_node_vocab = 60,
                input_node_embedd = 128,
                input_type = 'ragged', 
                # Output
                output_embedd = 'graph',
                output_use_bias = [True,False],
                output_dim = [25,1],
                output_activation = ['relu','sigmoid'],
                output_type = 'padded',
                #Model specific
                hidden_dim = 128,
                depth = 4,
                k = 0.3,
                score_initializer = 'ones',
                use_bias = True,
                activation = 'relu',
                is_sorted=False,
                has_unconnected=True,
                use_reconnect = True
        )


learning_rate_start = 1e-4
learning_rate_stop = 1e-5
epo = 500
epomin = 400
optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)

cbks = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction(learning_rate_start,learning_rate_stop,epomin,epo))
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())

trainlossall = []
testlossall = []
validlossall = []

epostep = 2

start = time.process_time()

hist = model.fit(xtrain, ytrain, 
          epochs=epo,
          batch_size=32,
          callbacks=[cbks],
          validation_freq=epostep,
          validation_data=(xval,yval),
          verbose=2
          )

trainlossall = hist.history['accuracy']
testlossall = hist.history['val_accuracy']

stop = time.process_time()
print("Print Time for taining: ",stop - start)

trainlossall =np.array(trainlossall)
testlossall = np.array(testlossall)

mae_valid = testlossall[-1]

#Plot loss vs epochs    
plt.figure()
plt.plot(np.arange(trainlossall.shape[0]),trainlossall,label='Training ACC',c='blue')
plt.plot(np.arange(epostep,epo+epostep,epostep),testlossall,label='Test ACC',c='red')
plt.scatter([trainlossall.shape[0]],[mae_valid],label="{0:0.4f} ".format(mae_valid),c='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Interaction Network Loss')
plt.legend(loc='upper right',fontsize='x-large')
plt.savefig('inorp_loss.png')
plt.show()