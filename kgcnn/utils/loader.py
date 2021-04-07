import numpy as np
import tensorflow as tf
import tensorflow.keras as ks


class GraphLoaderList(ks.utils.Sequence):
    """
    Data loader for graphs using tensorflow.keras.utils.Sequence.
    
    Note: Not tested yet.    
    """

    def __init__(self, label,
                 node,
                 edge_index,
                 edge,
                 state,
                 batch_size=32,
                 shuffle=False,
                 input_type="ragged",
                 output_embedd="graph",
                 output_type="padded"):
        """
        Pass data to loader via python list.

        Args:
            label (list): List or array of labels.
            node (list): List of node features.
            edge_index (list): List of edge_indices (i,j).
            edge (list): List of edge features. Defaults to None.
            state (list): List of graph specific state features. Defaults to None.
            batch_size (int): Size of the desired batch. Defaults to 32.
            shuffle (bool, optional): Shuffle data. Defaults to False.
            input_type (str, optional): The tensor type generated for model input. Defaults to "ragged".
            output_embedd (str, optional): If node or graph embedding is performed by model. Defaults to "graph".
            output_type (str, optional): The tensor type generated for model output. Defaults to "padded".

        Returns:
            None.

        """
        self.label_list = label
        self.node_list = node
        self.edge_index_list = edge_index
        self.edge_list = edge
        self.state_list = state
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_type = input_type
        self.output_type = output_type
        self.output_embedd = output_embedd

        self.indices = np.arange(len(label))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate index of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(batch_indices)

        # return batch_indexes
        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        # self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indices):
        """Generates data containing batch_size samples."""

        batch_label = [self.label_list[x] for x in batch_indices]
        batch_node = [self.node_list[x] for x in batch_indices]
        batch_edge_index = [self.edge_index_list[x] for x in batch_indices]
        batch_edge = [self.edge_list[x] for x in batch_indices]
        batch_state = [self.state_list[x] for x in batch_indices]

        if self.input_type == 'ragged':
            out_node = tf.RaggedTensor.from_row_lengths(np.concatenate(batch_node, axis=0),
                                                        np.array([len(x) for x in batch_node], dtype=np.int))
            out_edge = tf.RaggedTensor.from_row_lengths(np.concatenate(batch_edge, axis=0),
                                                        np.array([len(x) for x in batch_edge], dtype=np.int))
            out_edge_ind = tf.RaggedTensor.from_row_lengths(np.concatenate(batch_edge_index, axis=0),
                                                            np.array([len(x) for x in batch_edge_index], dtype=np.int))
            out_state = tf.constant(np.array(batch_state))
            if self.output_embedd == "graph":
                out_label = tf.constant(np.array(batch_label))
            else:
                out_temp = tf.RaggedTensor.from_row_lengths(np.concatenate(batch_label, axis=0),
                                                            np.array([len(x) for x in batch_label], dtype=np.int))
                if self.output_type == "ragged":
                    out_label = out_temp
                elif self.output_type == "padded":
                    out_label = out_temp.to_tensor()
                else:
                    raise NotImplementedError("Tensor output is not supported for", self.output_type)
        else:
            raise NotImplementedError("Tensor input is not supported for", self.input_type)

        out_x = [out_node, out_edge, out_edge_ind, out_state]
        out_y = out_label
        return out_x, out_y
