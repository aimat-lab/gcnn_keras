import keras as ks
from typing import Union
import numpy as np
from numpy.random import Generator, PCG64
import tensorflow as tf


def tf_disjoint_list_generator(
        graphs,
        inputs: list,
        outputs: list,
        assignment_to_id: list = None,
        assignment_of_indices: list = None,
        pos_batch_id: list = None,
        pos_subgraph_id: list = None,
        pos_count: list = None,
        batch_size=32,
        padded_disjoint=False,
        epochs=None,
        shuffle=True,
        seed=42
):
    dataset_size = len(graphs)
    data_index = np.arange(dataset_size)
    num_inputs = len(inputs)

    if len(assignment_to_id) < num_inputs:
        assignment_to_id = assignment_to_id + [None for _ in range(num_inputs-len(assignment_to_id))]
    if len(assignment_of_indices) < num_inputs:
        assignment_of_indices = assignment_of_indices + [None for _ in range(num_inputs-len(assignment_of_indices))]

    flag_batch_id = [None for _ in range(num_inputs)]
    for i, x in enumerate(pos_batch_id):
        flag_batch_id[x] = i

    flag_count = [None for _ in range(num_inputs)]
    for i, x in enumerate(pos_count):
        flag_count[x] = i

    flag_subgraph_id = [None for _ in range(num_inputs)]
    for i, x in enumerate(pos_subgraph_id):
        flag_subgraph_id[x] = i

    all_flags = [flag_batch_id, flag_count, flag_subgraph_id]
    is_attributes = [True if all([x[i] is None for x in all_flags]) else False for i in range(num_inputs)]

    if padded_disjoint:
        if epochs is None:
            raise ValueError("Requires number of epochs if `padded_disjoint=True` .")

    rng = Generator(PCG64(seed=seed))

    def generator():

        if shuffle:
            rng.shuffle(data_index)

        for batch_index in range(0, dataset_size, batch_size):
            idx = data_index[batch_index:batch_index + batch_size]
            graphs_batch = [graphs[i] for i in idx]

            out = [None for _ in range(num_inputs)]
            out_counts = [None for _ in range(num_inputs)]

            for i in range(num_inputs):
                if not is_attributes[i]:
                    continue

                array_list = [x[inputs[i]["name"]] for x in graphs_batch]
                if assignment_to_id[i] is None:
                    out[i] = np.array(array_list, dtype=inputs[i]["dtype"])
                else:
                    out[i] = np.concatenate(array_list, axis=0)
                    counts = np.array([len(x) for x in array_list], dtype="int64")
                    out_counts[i] = counts
                    ids = assignment_to_id[i]
                    if out[pos_count[ids]] is None:
                        out[pos_count[ids]] = counts
                    if out[pos_batch_id[ids]] is None:
                        out[pos_batch_id[ids]] = np.repeat(np.arange(len(array_list), dtype="int64"), repeats=counts)
                    if out[pos_subgraph_id[ids]] is None:
                        out[pos_subgraph_id[ids]] = np.concatenate([np.arange(x, dtype="int64") for x in counts], axis=0)

            # Indices
            for i in range(num_inputs):
                if assignment_of_indices[i] is not None:
                    edge_indices_flatten = out[i]
                    count_nodes = out_counts[assignment_of_indices[i]]
                    count_edges = out_counts[i]
                    node_splits = np.pad(np.cumsum(count_nodes), [[1, 0]])
                    offset_edge_indices = np.expand_dims(np.repeat(node_splits[:-1], count_edges), axis=-1)
                    disjoint_indices = edge_indices_flatten + offset_edge_indices
                    disjoint_indices = np.transpose(disjoint_indices)
                    out[i] = disjoint_indices

            if isinstance(outputs, list):
                out_y = []
                for k in range(len(outputs)):
                    array_list = [x[outputs[k]["name"]] for x in graphs_batch]
                    out_y.append(np.array(array_list, dtype=outputs[k]["dtype"]))
            else:
                out_y = np.array(
                    [x[outputs["name"]] for x in graphs_batch], dtype=outputs["dtype"])

            yield tuple(out), out_y

    input_spec = tuple([tf.TensorSpec(shape=tuple([None] + list(x["shape"])), dtype=x["dtype"]) for x in inputs])

    if isinstance(outputs, list):
        output_spec = tuple([tf.TensorSpec(shape=tuple([None] + list(x["shape"])), dtype=x["dtype"]) for x in outputs])
    else:
        output_spec = tf.TensorSpec(shape=tuple([None] + list(outputs["shape"])), dtype=outputs["dtype"])

    data_loader = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            input_spec,
            output_spec
        )
    )

    return data_loader
