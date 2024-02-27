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


def tf_dataset_disjoint_generator(
        graphs,
        inputs: Union[list, dict],
        assignment_to_id: Union[list, dict] = None,
        assignment_of_indices: Union[list, dict] = None,
        pos_batch_id: Union[list, dict] = None,
        pos_subgraph_id: Union[list, dict] = None,
        pos_count: Union[list, dict] = None,
        batch_size=32,
        epochs=None,
        padded_disjoint=False,
        shuffle=True,
        seed=42
):
    r"""Make a tensorflow dataset for disjoint graph loading.

    For the moment only IDs that have their values in inputs can be generated, as the value tensors of e.g. node
    or edge are used to generate batch IDs.

    Inputs is a list or dictionary of keras input layer configs. The names of the layers are linked to the properties
    in `graph` .

    With `assignment_to_id` and `assignment_of_indices` disjoint indices and attributes can be defined.
    Their IDs are marked with `pos_batch_id` etc. One must use a name or index for each general split, since for
    example edge IDs can be used for edge indices, edge attributes and edge relation tensors at the same time.
    Therefore, one batch ID for edges is enough. One could however assign as many as IDs as there are disjoint
    graph properties in `graph` .

    Args:
        graphs: List of dictionaries with named graph properties.
        inputs: List or dict of keras input layer configs.
        assignment_to_id: Assignment of if inputs to disjoint properties to IDs.
        assignment_of_indices: Assignment of inputs (if they are indices) to their reference.
        pos_batch_id: Position or name of batch IDs.
        pos_subgraph_id: Position or name of batch IDs.
        pos_count: Position or name of batch IDs.
        batch_size: Batch size.
        epochs: Expected number of epochs. Only required for padded disjoint.
        padded_disjoint: If padded disjoint tensors should be generated.
        shuffle: Whether to shuffle each epoch.
        seed: Seed for shuffle.

    Returns:
        tf.data.Dataset: Tensorflow dataset to load disjoint graphs.
    """
    # Stats on the required dataset.
    dataset_size = len(graphs)
    data_index = np.arange(dataset_size)
    num_inputs = len(inputs)

    # Check input information for outputspec.
    is_single_input = False
    is_list_input = False
    if isinstance(inputs, list):
        is_list_input = True
        output_spec = tuple([tf.TensorSpec(shape=tuple([None] + list(x["shape"])), dtype=x["dtype"]) for x in inputs])
    elif isinstance(inputs, dict):
        if "shape" in inputs and "dtype" in inputs:
            output_spec = tf.TensorSpec(shape=tuple([None] + list(inputs["shape"])), dtype=inputs["dtype"])
            inputs = {0: inputs}
            is_single_input = True
            num_inputs = 1
        else:
            output_spec = dict(
                {i: tf.TensorSpec(shape=tuple([None] + list(x["shape"])), dtype=x["dtype"]) for i, x in inputs.items()})
    else:
        raise ValueError("Inputs must be list or dict of keras input layer kwargs.")

    # We use a dict for both list and dict input.
    def _convert_to_dict(container_to_check):
        if container_to_check is None:
            return {}
        if isinstance(container_to_check, (list, tuple)):
            return {i: x for i, x in enumerate(container_to_check)}
        if not isinstance(container_to_check, dict):
            raise ValueError("Must be dict or list for mapping and containers.")
        return container_to_check

    inputs = _convert_to_dict(inputs)
    assignment_to_id = _convert_to_dict(assignment_to_id)
    assignment_of_indices = _convert_to_dict(assignment_of_indices)
    pos_batch_id = _convert_to_dict(pos_batch_id)
    pos_subgraph_id = _convert_to_dict(pos_subgraph_id)
    pos_count = _convert_to_dict(pos_count)

    # Fill assignments with Nones if they are not used for input.
    if len(assignment_to_id) < num_inputs:
        for key, values in inputs.items():
            if key not in assignment_to_id.keys():
                assignment_to_id[key] = None
    if len(assignment_of_indices) < num_inputs:
        for key, values in inputs.items():
            if key not in assignment_of_indices.keys():
                assignment_of_indices[key] = None

    flag_batch_id = {i: None for i in inputs.keys()}
    for i, x in pos_batch_id.items():
        flag_batch_id[x] = i

    flag_count = {i: None for i in inputs.keys()}
    for i, x in pos_count.items():
        flag_count[x] = i

    flag_subgraph_id = {i: None for i in inputs.keys()}
    for i, x in pos_subgraph_id.items():
        flag_subgraph_id[x] = i

    all_flags = [flag_batch_id, flag_count, flag_subgraph_id]
    is_attributes = {i: True if all([x[i] is None for x in all_flags]) else False for i in inputs.keys()}

    max_size = {i: [] if assignment_to_id[i] is not None else None for i in inputs.keys()}
    total_max = {i: [] if assignment_to_id[i] is not None else None for i in inputs.keys()}

    # We can check the maximum batch size at the beginning or just have a maximum batch size for each epoch.
    if padded_disjoint:
        if epochs is None:
            raise ValueError("Requires number of epochs if `padded_disjoint=True` .")

        for i in inputs.keys():
            if assignment_to_id[i] is None:
                continue
            len_list = [len(x[inputs[i]["name"]]) for x in graphs]
            total_max[i] = max(len_list)

        rng = Generator(PCG64(seed=seed))

        for epoch in range(epochs):
            max_size_epoch = {i: [] if assignment_to_id[i] is not None else None for i in inputs.keys()}
            if shuffle:
                rng.shuffle(data_index)
            for batch_index in range(0, dataset_size, batch_size):
                idx = data_index[batch_index:batch_index + batch_size]
                graphs_batch = [graphs[i] for i in idx]
                for i in inputs.keys():
                    if assignment_to_id[i] is None:
                        continue
                    len_list = [len(x[inputs[i]["name"]]) for x in graphs_batch]
                    max_length = sum(len_list)
                    max_size_epoch[i].append(max_length)
            for i, x in max_size_epoch.items():
                if x is not None:
                    max_size[i].append(max(x))
        max_size = {i: max(x) if x is not None else None for i, x in max_size.items()}

        module_logger.info("Max of graph: %s." % total_max)
        module_logger.info("Padded max of disjoint: %s." % [
            x/batch_size if x is not None else None for x in max_size.values()])

    data_index = np.arange(dataset_size)
    rng = Generator(PCG64(seed=seed))

    def generator():

        if shuffle:
            rng.shuffle(data_index)

        for batch_index in range(0, dataset_size, batch_size):
            idx = data_index[batch_index:batch_index + batch_size]
            graphs_batch = [graphs[i] for i in idx]

            out = {i: None for i in inputs.keys()}
            out_counts = {i: None for i in inputs.keys()}

            for i in inputs.keys():
                if not is_attributes[i]:
                    continue

                array_list = [x[inputs[i]["name"]] for x in graphs_batch]
                if assignment_to_id[i] is None:
                    values = np.array(array_list, dtype=inputs[i]["dtype"])
                    if padded_disjoint:
                        values = pad_at_axis(values, (1, 0), axis=0)
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

                    if ids in pos_count:
                        if out[pos_count[ids]] is None:
                            out[pos_count[ids]] = counts
                    if ids in pos_batch_id:
                        if out[pos_batch_id[ids]] is None:
                            out[pos_batch_id[ids]] = np.repeat(
                                np.arange(len(counts), dtype="int64"), repeats=counts)
                    if ids in pos_subgraph_id:
                        if out[pos_subgraph_id[ids]] is None:
                            out[pos_subgraph_id[ids]] = np.concatenate(
                                [np.arange(x, dtype="int64") for x in counts], axis=0)

            # Indices
            for i in inputs.keys():
                if assignment_of_indices[i] is not None:
                    edge_indices_flatten = out[i]
                    count_nodes = out_counts[assignment_of_indices[i]]
                    count_edges = out_counts[i]
                    node_splits = np.pad(np.cumsum(count_nodes), [[1, 0]])
                    offset_edge_indices = np.expand_dims(np.repeat(node_splits[:-1], count_edges), axis=-1)
                    disjoint_indices = edge_indices_flatten + offset_edge_indices
                    disjoint_indices = np.transpose(disjoint_indices)
                    out[i] = disjoint_indices

            # Match output container
            if is_list_input:
                out = tuple([out[i] for i in range(num_inputs)])
            if is_single_input:
                out = out[0]

            yield out

    data_loader = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_spec
    )

    return data_loader
