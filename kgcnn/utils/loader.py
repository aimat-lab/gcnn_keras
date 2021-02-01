class GraphLoader(ks.utils.Sequence):
    """
    Data loader for graphs using tensorflow.keras.utils.Sequence.
    
    """
    def __init__(self,label_list,node_list,edge_index_list,edge_list=None,state_list=None,batch_size=32, shuffle=False, label_type= "graph"):
        """
        Initialization with data reference.
        
        Args:
            label_list : (list)
            node_list : (list)
            edge_index_list : (list)
            edge_list : (list)
            state_list : (list)
            batch_size=32 : (int)
            Shuffle=False : (bool)
        """
        self.label_list = label_list
        self.node_list = node_list
        self.edge_index_list = edge_index_list
        self.edge_list = edge_list
        self.state_list = state_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_type = label_type

        self.indices = np.arange(len(list_labels))
        self.on_epoch_end()        
        
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate index of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_indices)
        
        #return batch_indexes
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        #self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indices):
        """Generates data containing batch_size samples"""
        batch_label_list = [self.label_list[x] for x in batch_indices]
        batch_node_list = [self.node_list[x] for x in batch_indices]
        batch_edge_index_list = [self.edge_index_list[x] for x in batch_indices]
        batch_edge_list = None
        batch_state_list = None
        
        # Make conversion here.
        # TODO
        
        out_x = None
        out_y = None
        return out_x,out_y
