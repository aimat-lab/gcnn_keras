import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from kgcnn.data.qm.qm9 import qm9_graph
from kgcnn.literature.Megnet import getmodelMegnet,softplus2
from kgcnn.utils.learning import lr_lin_reduction


# Download Dataset
qm9_data = qm9_graph()
y_data = qm9_data[0][:,7]*27.2114  #select LUMO in eV
x_data = qm9_data[1:]

#Scale output
y_mean = np.mean(y_data)
y_data = (np.expand_dims(y_data,axis=-1)-y_mean)  
data_unit = 'eV'

#Make test/train split
VALSIZE = 100
TRAINSIZE = 2000
print("Training Size:",TRAINSIZE," Validation Size:",VALSIZE )
inds = np.arange(len(y_data))
inds = shuffle(inds)
ind_val = inds[:VALSIZE ]
ind_train = inds[VALSIZE:(VALSIZE + TRAINSIZE)]

# Select train/test data
xtrain = [[x[i] for i in ind_train] for x in x_data]
ytrain = y_data[ind_train]
xval = [[x[i] for i in ind_val] for x in x_data]
yval = y_data[ind_val]

def make_ragged(inlist):
    return tf.RaggedTensor.from_row_lengths(np.concatenate(inlist,axis=0), np.array([len(x) for x in inlist],dtype=np.int))

#Make ragged graph tensors plus normal tensor for graph state
xval = [make_ragged(x) for x in xval[:3]] + [tf.constant(xval[3])]
xtrain = [make_ragged(x) for x in xtrain[:3]] + [tf.constant(xtrain[3])]


model =  getmodelMegnet(
                    # Input
                    input_node_shape = [None],
                    input_edge_shape = [None,20],
                    input_state_shape = [1],
                    input_node_vocab = 10,
                    input_node_embedd = 16,
                    input_edge_embedd = 16,
                    input_type = 'ragged', 
                    # Output
                    output_embedd = 'graph', #Only graph possible for megnet
                    output_use_bias = [True,True,True],
                    output_dim = [32,16,1],
                    output_activation = ['softplus2','softplus2','linear'],
                    output_type = 'padded',
                    #Model specs
                    is_sorted = True,
                    has_unconnected = False,
                    nblocks = 3,
                    n1= 64,
                    n2 = 32,
                    n3= 16,
                    set2set_dim = 16,
                    use_bias = True,
                    act = 'softplus2',
                    l2_coef = None,
                    has_ff = True,
                    dropout = None,
                    dropout_on_predict = False,
                    use_set2set = True,
                    npass= 3,
                    set2set_init = '0',
                    set2set_pool = "sum"
                    )


learning_rate_start = 0.5e-3
learning_rate_stop = 1e-5
epo = 500
epomin = 400
optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)

cbks = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction(learning_rate_start,learning_rate_stop,epomin,epo))
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])
print(model.summary())

trainlossall = []
testlossall = []
validlossall = []

epostep = 10

start = time.process_time()

hist = model.fit(xtrain, ytrain, 
          epochs=epo,
          batch_size=64,
          callbacks=[cbks],
          validation_freq=epostep,
          validation_data=(xval,yval),
          verbose=2
          )

trainlossall = hist.history['mean_absolute_error']
testlossall = hist.history['val_mean_absolute_error']

stop = time.process_time()
print("Print Time for taining: ",stop - start)

trainlossall =np.array(trainlossall)
testlossall = np.array(testlossall)

mae_valid = np.mean(np.abs(yval-model.predict(xval)))

#Plot loss vs epochs    
plt.figure()
plt.plot(np.arange(trainlossall.shape[0]),trainlossall,label='Training Loss',c='blue')
plt.plot(np.arange(epostep,epo+epostep,epostep),testlossall,label='Test Loss',c='red')
plt.scatter([trainlossall.shape[0]],[mae_valid],label="{0:0.4f} ".format(mae_valid)+"["+data_unit +"]",c='red')
plt.xlabel('Epochs')
plt.ylabel('Loss ' + "["+data_unit +"]")
plt.title('Megnet Loss')
plt.legend(loc='upper right',fontsize='x-large')
plt.savefig('megnet_loss.png')
plt.show()

#Predicted vs Actual    
preds = model.predict(xval)
plt.figure()
plt.scatter(preds+y_mean, yval+y_mean, alpha=0.3,label="MAE: {0:0.4f} ".format(mae_valid)+"["+data_unit +"]")
plt.plot(np.arange(np.amin(yval), np.amax(yval),0.05), np.arange(np.amin(yval), np.amax(yval),0.05), color='red')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.legend(loc='upper left',fontsize='x-large')
plt.savefig('megnet_predict.png')
plt.show()