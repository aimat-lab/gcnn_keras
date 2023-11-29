import keras as ks
import numpy as np
import tensorflow as tf


def experimental_tf_disjoint_list_generator(graphs,
                                            inputs,
                                            outputs,
                                            has_nodes=True,
                                            has_edges=True,
                                            has_graph_state=False,
                                            batch_size=32,
                                            shuffle=True):
    def generator():
        dataset_size = len(graphs)
        data_index = np.arange(dataset_size)

        if shuffle:
            np.random.shuffle(data_index)

        for batch_index in range(0, dataset_size, batch_size):
            idx = data_index[batch_index:batch_index + batch_size]
            graphs_batch = [graphs[i] for i in idx]

            batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges = [None for _ in range(6)]
            out = []
            inputs_pos = 0
            for j in range(int(has_nodes)):
                array_list = [x[inputs[inputs_pos]["name"]] for x in graphs_batch]
                out.append(np.concatenate(array_list, axis=0))
                inputs_pos += 1
                if j == 0:
                    count_nodes = np.array([len(x) for x in array_list], dtype="int64")
                    batch_id_node = np.repeat(np.arange(len(array_list), dtype="int64"), repeats=count_nodes)
                    node_id = np.concatenate([np.arange(x, dtype="int64") for x in count_nodes], axis=0)

            for j in range(int(has_edges)):
                array_list = [x[inputs[inputs_pos]["name"]] for x in graphs_batch]
                out.append(np.concatenate(array_list, axis=0, dtype=inputs[inputs_pos]["dtype"]))
                inputs_pos += 1

            for j in range(int(has_graph_state)):
                array_list = [x[inputs[inputs_pos]["name"]] for x in graphs_batch]
                out.append(np.array(array_list, dtype=inputs[inputs_pos]["dtype"]))
                inputs_pos += 1

            # Indices
            array_list = [x[inputs[inputs_pos]["name"]] for x in graphs_batch]
            count_edges = np.array([len(x) for x in array_list], dtype="int64")
            batch_id_edge = np.repeat(np.arange(len(array_list), dtype="int64"), repeats=count_edges)
            edge_id = np.concatenate([np.arange(x, dtype="int64") for x in count_edges], axis=0)
            edge_indices_flatten = np.concatenate(array_list, axis=0)

            node_splits = np.pad(np.cumsum(count_nodes), [[1, 0]])
            offset_edge_indices = np.expand_dims(np.repeat(node_splits[:-1], count_edges), axis=-1)
            disjoint_indices = edge_indices_flatten + offset_edge_indices
            disjoint_indices = np.transpose(disjoint_indices)
            out.append(disjoint_indices)

            out = out + [batch_id_node, batch_id_edge, node_id, edge_id, count_nodes, count_edges]

            if isinstance(outputs, list):
                out_y = []
                for k in range(len(outputs)):
                    array_list = [x[outputs[k]["name"]] for x in graphs_batch]
                    out_y.append(np.array(array_list, dtype=outputs[k]["dtype"]))
            elif isinstance(outputs, dict):
                out_y = np.array(
                    [x[outputs["name"]] for x in graphs_batch], dtype=outputs["dtype"])
            else:
                raise ValueError()

            yield tuple(out), out_y

    input_spec = tuple([tf.TensorSpec(shape=tuple([None] + list(x["shape"])), dtype=x["dtype"]) for x in inputs])

    if isinstance(outputs, list):
        output_spec = tuple([tf.TensorSpec(shape=tuple([None] + list(x["shape"])), dtype=x["dtype"]) for x in outputs])
    elif isinstance(outputs, dict):
        output_spec = tf.TensorSpec(shape=tuple([None] + list(outputs["shape"])), dtype=outputs["dtype"])
    else:
        raise ValueError()

    data_loader = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            input_spec,
            output_spec
        )
    )

    return data_loader
