import keras as ks
import logging
from typing import Union
import numpy as np
from numpy.random import Generator, PCG64
import tensorflow as tf


# Module logger
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def pad_at_axis(x, pad_width, axis=0, **kwargs):
    pads = [(0, 0) for _ in range(len(x.shape))]
    pads[axis] = pad_width
    return np.pad(x, pad_width=pads, **kwargs)


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
        epochs=None,
        padded_disjoint=False,
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

    max_size = [[] if assignment_to_id[i] is not None else None for i in range(num_inputs)]
    total_max = [[] if assignment_to_id[i] is not None else None for i in range(num_inputs)]

    # We can check the maximum batch size at the beginning or just have a maximum batch size for each epoch.
    if padded_disjoint:
        if epochs is None:
            raise ValueError("Requires number of epochs if `padded_disjoint=True` .")

        for i in range(num_inputs):
            if assignment_to_id[i] is None:
                continue
            len_list = [len(x[inputs[i]["name"]]) for x in graphs]
            total_max[i] = max(len_list)

        rng = Generator(PCG64(seed=seed))

        for epoch in range(epochs):
            max_size_epoch = [[] if assignment_to_id[i] is not None else None for i in range(num_inputs)]
            if shuffle:
                rng.shuffle(data_index)
            for batch_index in range(0, dataset_size, batch_size):
                idx = data_index[batch_index:batch_index + batch_size]
                graphs_batch = [graphs[i] for i in idx]
                for i in range(num_inputs):
                    if assignment_to_id[i] is None:
                        continue
                    len_list = [len(x[inputs[i]["name"]]) for x in graphs_batch]
                    max_length = sum(len_list)
                    max_size_epoch[i].append(max_length)
            for i, x in enumerate(max_size_epoch):
                if x is not None:
                    max_size[i].append(max(x))
        max_size = [max(x) if x is not None else None for x in max_size]

        module_logger.info("Max of graph: %s." % total_max)
        module_logger.info("Padded max of disjoint: %s." % [
            x/batch_size if x is not None else None for x in max_size])

    data_index = np.arange(dataset_size)
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
                    values = np.array(array_list, dtype=inputs[i]["dtype"])
                    if padded_disjoint:
                        out = pad_at_axis(values, (1, 0), axis=0)
                    out[i] = values
                else:
                    values = np.concatenate(array_list, axis=0)
                    counts = np.array([len(x) for x in array_list], dtype="int64")
                    ids = assignment_to_id[i]

                    if not padded_disjoint:
                        out[i] = values
                        out_counts[i] = counts
                    else:
                        len_values = len(values)
                        num_pad_required = max_size[i] - len_values + 1
                        values = pad_at_axis(values, (num_pad_required, 0), axis=0)
                        out[i] = values
                        counts = np.concatenate([np.array([num_pad_required], dtype=counts.dtype), counts], axis=0)
                        out_counts[i] = counts

                    if out[pos_count[ids]] is None:
                        out[pos_count[ids]] = counts
                    if out[pos_batch_id[ids]] is None:
                        out[pos_batch_id[ids]] = np.repeat(
                            np.arange(len(counts), dtype="int64"), repeats=counts)
                    if out[pos_subgraph_id[ids]] is None:
                        out[pos_subgraph_id[ids]] = np.concatenate(
                            [np.arange(x, dtype="int64") for x in counts], axis=0)

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
