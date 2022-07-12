import logging
import numpy as np
import tensorflow as tf
import networkx as nx
import pandas as pd
import os
import re
from kgcnn.data.adapter import GraphMethodsAdapter
from kgcnn.data.utils import save_pickle_file, load_pickle_file, ragged_tensor_from_nested_numpy

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class GraphDict(dict, GraphMethodsAdapter):
    r"""Dictionary container to store graph information in tensor-form. At the moment only numpy-arrays are supported.
    The naming convention is not restricted. The class is supposed to be handled just as a python dictionary.
    In addition, :obj:`assign_property` and :obj:`obtain_property` handles `None` values and cast into tensor format,
    when assigning a named value.

    Graph operations that modify edges or sort indices are methods of this class supported by
    :obj:`kgcnn.data.adapter.GraphMethodsAdapter`.
    Note that the graph-tensors name must follow a standard-convention or be provided to member functions
    (see documentation of :obj:`kgcnn.data.adapter.GraphMethodsAdapter`).

    .. code-block:: python

        import numpy as np
        from kgcnn.data.base import GraphDict
        g = GraphDict({"edge_indices": np.array([[1, 0], [0, 1]]), "edge_labels": np.array([[-1], [1]])})
        g.add_edge_self_loops().sort_edge_indices()  # from GraphMethodsAdapter
        print(g)
    """

    # Implementation details: Inherits from python-dict at the moment but can be changed if this causes problems,
    # alternatives would be: collections.UserDict or collections.abc.MutableMapping

    def __init__(self, sub_dict: dict = None):
        r"""Initialize a new :obj:`GraphDict` instance.

        Args:
            sub_dict: Dictionary or key-value pair of numpy arrays.
        """
        self._tensor_conversion = np.array
        if sub_dict is None:
            sub_dict = {}
        elif isinstance(sub_dict, (dict, list)):
            in_dict = dict(sub_dict)
            sub_dict = {key: self._tensor_conversion(value) for key, value in in_dict.items()}
        elif isinstance(sub_dict, GraphDict):
            sub_dict = {key: self._tensor_conversion(value) for key, value in sub_dict.items()}
        super(GraphDict, self).__init__(sub_dict)

    def to_dict(self) -> dict:
        """Returns a python-dictionary of self. Does not copy values.

        Returns:
            dict: Dictionary of graph tensor objects.
        """
        return {key: value for key, value in self.items()}

    def from_networkx(self, graph,
                      node_number: str = "node_number",
                      edge_indices: str = "edge_indices",
                      node_attributes: str = None,
                      edge_attributes: str = None,
                      node_labels: str = None):
        r"""Convert a networkx graph instance into a dictionary of graph-tensors. The networkx graph is always converted
        into integer node labels. The former node IDs can be hold in :obj:`node_labels`. Furthermore, node or edge
        data can be cast into attributes via :obj:`node_attributes` and :obj:`edge_attributes`.

        Args:
            graph (nx.Graph): A networkx graph instance to convert.
            node_number (str): The name that the node numbers are assigned to. Default is "node_number".
            edge_indices (str): The name that the edge indices are assigned to. Default is "edge_indices".
            node_attributes (str, list): Name of node attributes to add from node data. Can also be a list of names.
                Default is None.
            edge_attributes (str, list): Name of edge attributes to add from edge data. Can also be a list of names.
                Default is None.
            node_labels (str): Name of the labels of nodes to store former node IDs into. Default is None.

        Returns:
            self.
        """
        assert node_labels is None or isinstance(node_labels, str), "Please provide name of node labels or `None`"
        graph_int = nx.convert_node_labels_to_integers(graph, label_attribute=node_labels)
        graph_size = len(graph_int)

        def _attr_to_list(attr):
            if attr is None:
                attr = []
            elif isinstance(attr, str):
                attr = [attr]
            if not isinstance(attr, list):
                raise TypeError("Attribute name is neither list nor string.")
            return attr

        # Loop over nodes in graph.
        node_attr = _attr_to_list(node_attributes)
        if node_labels is not None:
            node_attr += [node_labels]
        node_attr_dict = {x: [None]*graph_size for x in node_attr}
        nodes_id = []
        for i, x in enumerate(graph_int.nodes.data()):
            nodes_id.append(x[0])
            for d in node_attr:
                if d not in x[1]:
                    raise KeyError("Node does not have property %s" % d)
                node_attr_dict[d][i] = x[1][d]

        edge_id = []
        edges_attr = _attr_to_list(edge_attributes)
        edges_attr_dict = {x: [None]*graph.number_of_edges() for x in edges_attr}
        for i, x in enumerate(graph_int.edges.data()):
            edge_id.append(x[:2])
            for d in edges_attr:
                if d not in x[2]:
                    raise KeyError("Edge does not have property %s" % d)
                edges_attr_dict[d][i] = x[2][d]

        # Storing graph tensors in self.
        self.assign_property(node_number, self._tensor_conversion(nodes_id))
        self.assign_property(edge_indices, self._tensor_conversion(edge_id))
        for key, value in node_attr_dict.items():
            self.assign_property(key, self._tensor_conversion(value))
        for key, value in edges_attr_dict.items():
            self.assign_property(key, self._tensor_conversion(value))
        return self

    def to_networkx(self, edge_indices: str = "edge_indices"):
        """Function draft to make a networkx graph. No attributes or data is supported at the moment.

        Args:
            edge_indices (str): Name of edge index tensors to make graph with. Default is "edge_indices".

        Returns:
            nx.DiGraph: Directed networkx graph instance.
        """
        graph = nx.DiGraph()
        graph.add_edges_from(self.obtain_property(edge_indices))
        return graph

    def assign_property(self, key: str, value):
        r"""Add a named property as key, value pair to self. If the value is `None`, nothing is done.
        Similar to assign-item default method :obj:`__setitem__`, but ignores `None` values and casts to tensor.

        Args:
            key (str): Name of the graph tensor to add to self.
            value: Array or tensor value to add. Can also be None.

        Returns:
            None.
        """
        if value is not None:
            self.update({key: self._tensor_conversion(value)})

    def obtain_property(self, key: str):
        """Get tensor item by name. If key is not found, `None` is returned.

        Args:
            key (str): Name of the key to get value for.

        Returns:
            self[key].
        """
        if key in self:
            return self[key]
        return None

    def find_graph_properties(self, name_props: str) -> list:
        r"""Search for properties in self. This includes a list of possible names or a pattern-matching of a single
        string.

        Args:
            name_props (str, list): Pattern matching string or list of strings to search for

        Returns:
            list: List of names in self that match :obj:`name_props`.
        """
        if name_props is None:
            return []
        elif isinstance(name_props, str):
            match_props = []
            for x in self:
                if re.match(name_props, x):
                    if re.match(name_props, x).group() == x:
                        match_props.append(x)
            return match_props
        elif isinstance(name_props, (list, tuple)):
            return [x for x in name_props if x in self]
        raise TypeError("Can not find keys of properties for input type %s" % name_props)

    def assert_has_key(self, key: str, raise_error: bool = False):
        """Check if the property is found in self.

        Args:
            key (str): Name of property that must be defined.
            raise_error (bool): Whether to raise error. Default is False.

        Returns:
            bool: Key is valid.
        """
        if key not in self or self.obtain_property(key) is None:
            if raise_error:
                raise ValueError("Can not use '%s', as it is not found." % key)
            return False
        return True


GraphNumpyContainer = GraphDict


class MemoryGraphList:
    r"""Class to store a list of graph dictionaries in memory. Contains a python list as property :obj:`_list`.
    The graph properties are defined by tensor-like numpy arrays for indices, attributes, labels, symbol etc. .
    They are distributed from of a list of numpy arrays to each graph in the list via :obj:`assign_property`.

    A list of numpy arrays can also be passed to class instances that have a reserved prefix, which is essentially
    a shortcut to :obj:`assign_property` and must match the length of the list.
    Prefix are `node_`, `edge_`, `graph_` `range_` and `angle_` for their node, edge and graph properties, respectively.
    The range-attributes and range-indices are just like edge-indices but refer to a geometric annotation.
    This allows to have geometric range-connections and topological edges separately.
    The prefix 'range' is synonym for a geometric edge.

    .. code-block:: python

        import numpy as np
        from kgcnn.data.base import MemoryGraphList
        data = MemoryGraphList()
        data.empty(1)
        data.assign_property("edge_indices", [np.array([[0, 1], [1, 0]])])
        data.assign_property("node_labels", [np.array([[0], [1]])])
        print(data.obtain_property("edge_indices"))
        data.assign_property("node_coordinates", [np.array([[1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])])
        print(data.obtain_property("node_coordinates"))
        data.map_list("set_range", max_distance=1.5, max_neighbours=10, self_loops=False)
    """

    def __init__(self, input_list: list = None):
        r"""Initialize an empty :obj:`MemoryGraphList` instance. If you want to expand the list or
        namespace of accepted reserved graph prefix identifier, you can expand :obj:`_reserved_graph_property_prefix`.

        Args:
            input_list (list, MemoryGraphList): A list or :obj:`MemoryGraphList` of :obj:`GraphDict` items.
        """
        self._list = []
        self._reserved_graph_property_prefix = ["node_", "edge_", "graph_", "range_", "angle_"]
        self.logger = module_logger
        if input_list is None:
            input_list = []
        if isinstance(input_list, list):
            self._list = [GraphDict(x) for x in input_list]
        if isinstance(input_list, MemoryGraphList):
            self._list = [GraphDict(x) for x in input_list._list]

    def assign_property(self, key, value):
        if value is None:
            # We could also here remove the key from all graphs.
            return self
        if not isinstance(value, list):
            raise TypeError("Expected type 'list' to assign graph properties.")
        if len(self._list) == 0:
            self.empty(len(value))
        if len(self._list) != len(value):
            raise ValueError("Can only store graph attributes from list with same length.")
        for i, x in enumerate(value):
            self._list[i].assign_property(key, x)
        return self

    def obtain_property(self, key):
        r"""Returns a list with the values of all the graphs defined for the string property name `key`. If none of
        the graphs in the list have this property, returns None.

        Args:
            key (str): The string name of the property to be retrieved for all the graphs contained in this list
        """
        # "_list" is a list of GraphDicts, which means "prop_list" here will be a list of all the property
        # values for teach of the graphs which make up this list.
        prop_list = [x.obtain_property(key) for x in self._list]

        # If a certain string property is not set for a GraphDict, it will still return None. Here we check:
        # If all the items for our given property name are None then we know that this property is generally not
        # defined for any of the graphs in the list.
        if all([x is None for x in prop_list]):
            self.logger.warning("Property %s is not set on any graph." % key)
            return None

        return prop_list

    def __setattr__(self, key, value):
        """Setter that intercepts reserved attributes and stores them in the list of graph containers."""
        if not hasattr(self, "_reserved_graph_property_prefix") or not hasattr(self, "_list"):
            return super(MemoryGraphList, self).__setattr__(key, value)
        if any([x == key[:len(x)] for x in self._reserved_graph_property_prefix]):
            module_logger.warning(
                "Reserved properties are deprecated and will be removed. Please use `assign_property()`.")
            self.assign_property(key, value)
        else:
            return super(MemoryGraphList, self).__setattr__(key, value)

    def __getattribute__(self, key):
        """Getter that retrieves a list of properties from graph containers."""
        if key in ["_reserved_graph_property_prefix", "_list"]:
            return super(MemoryGraphList, self).__getattribute__(key)
        if any([x == key[:len(x)] for x in self._reserved_graph_property_prefix]):
            module_logger.warning(
                "Reserved properties are deprecated and will be removed. Please use `obtain_property()`.")
            return self.obtain_property(key)
        else:
            return super().__getattribute__(key)

    def __len__(self):
        """Return the current length of this instance."""
        return len(self._list)

    def __getitem__(self, item):
        # Does not make a copy of the data, as a python list does.
        if isinstance(item, int):
            return self._list[item]
        new_list = MemoryGraphList()
        if isinstance(item, slice):
            return new_list._set_internal_list(self._list[item])
        if isinstance(item, list):
            return new_list._set_internal_list([self._list[int(i)] for i in item])
        if isinstance(item, np.ndarray):
            return new_list._set_internal_list([self._list[int(i)] for i in item])
        raise TypeError("Unsupported type for MemoryGraphList items.")

    def _set_internal_list(self, value: list):
        if not isinstance(value, list):
            raise TypeError("Must set list for MemoryGraphList.")
        self._list = value
        return self

    def __setitem__(self, key, value):
        if not isinstance(value, GraphDict):
            raise TypeError("Require a GraphDict as list item.")
        self._list[key] = value

    def clear(self):
        self._list = []

    def empty(self, length: int):
        if length is None:
            return self
        if length < 0:
            raise ValueError("Length of empty list must be >=0.")
        self._list = [GraphDict() for _ in range(length)]
        return self

    @property
    def length(self):
        return len(self._list)

    @length.setter
    def length(self, value: int):
        raise ValueError("Can not set length. Please use 'empty()' to initialize an empty list.")

    def _to_tensor(self, item, make_copy=True):
        if not make_copy:
            self.logger.warning("At the moment always a copy is made for tensor().")
        props = self.obtain_property(item["name"])  # Will be list.
        is_ragged = item["ragged"] if "ragged" in item else False
        if is_ragged:
            return ragged_tensor_from_nested_numpy(props)
        else:
            return tf.constant(np.array(props))

    def tensor(self, items, make_copy=True):
        if isinstance(items, dict):
            return self._to_tensor(items, make_copy=make_copy)
        elif isinstance(items, (tuple, list)):
            return [self._to_tensor(x, make_copy=make_copy) for x in items]
        else:
            raise TypeError("Wrong type, expected e.g. [{'name': 'edge_indices', 'ragged': True}, {...}, ...]")

    def map_list(self, method, **kwargs):
        for x in self._list:
            getattr(x, method)(**kwargs)
        return self

    def clean(self, inputs: list):
        r"""Given a list of property names, this method removes all elements from the internal list of
        `GraphDict` items, which do not define at least one of those properties. Meaning, only those graphs remain in
        the list which definitely define all properties specified by :obj:`inputs`.

        Args:
            inputs (list): A list of strings, where each string is supposed to be a property name, which the graphs
                in this list may possess.

        Returns:
            invalid_graphs (np.ndarray): A list of graph indices that do not have the required properties and which
                have been removed.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        invalid_graphs = []
        for item in inputs:
            if isinstance(item, dict):
                item_name = item["name"]
            else:
                item_name = item
            props = self.obtain_property(item_name)
            if props is None:
                self.logger.warning("Can not clean property %s as it was not assigned to any graph." % item)
                continue
            for i, x in enumerate(props):
                # If property is neither list nor np.array
                if x is None or not hasattr(x, "__getitem__"):
                    self.logger.info("Property %s is not defined for graph %s." % (item_name, i))
                    invalid_graphs.append(i)
                elif not isinstance(x, np.ndarray):
                    self.logger.info("Property %s is not a numpy array for graph %s." % (item_name, i))
                    invalid_graphs.append(i)
                elif len(x.shape) > 0:
                    if len(x) <= 0:
                        self.logger.info("Property %s is empty list for graph %s." % (item_name, i))
                        invalid_graphs.append(i)
        invalid_graphs = np.unique(np.array(invalid_graphs, dtype="int"))
        invalid_graphs = np.flip(invalid_graphs)  # Need descending order
        if len(invalid_graphs) > 0:
            self.logger.warning("Found invalid graphs for properties. Removing graphs %s." % invalid_graphs)
        else:
            self.logger.info("No invalid graphs for assigned properties found.")
        # Remove from the end via pop().
        for i in invalid_graphs:
            self._list.pop(int(i))
        return invalid_graphs


class MemoryGraphDataset(MemoryGraphList):
    r"""Dataset class for lists of graph tensor properties that can be cast into the :obj:`tf.RaggedTensor` class.
    The graph list is expected to only store numpy arrays in place of the each node or edge information!

    .. note::
        Each graph attribute is expected to be a python list or iterable object containing numpy arrays.
        For example, the special attribute of :obj:`edge_indices` is expected to be a list of arrays of
        shape `(Num_edges, 2)` with the indices of node connections.
        The node attributes in :obj:`node_attributes` are numpy arrays of shape `(Num_nodes, Num_features)`.

    The Memory Dataset class inherits from :obj:`MemoryGraphList` and has further information
    about a location on disk, i.e. a file directory and a file name as well as a name of the dataset.

    .. code-block:: python

        from kgcnn.data.base import MemoryGraphDataset
        dataset = MemoryGraphDataset(data_directory="", dataset_name="Example")
        dataset.assign_property("edge_indices", [np.array([[1, 0], [0, 1]])])
        dataset.assign_property("edge_labels", [np.array([[0], [1]])])

    The file directory and file name are used in child classes and in :obj:`save` and :obj:`load`.
    """

    fits_in_memory = True

    def __init__(self,
                 data_directory: str = None,
                 dataset_name: str = None,
                 file_name: str = None,
                 file_directory: str = None,
                 verbose: int = 10,
                 ):
        r"""Initialize a base class of :obj:`MemoryGraphDataset`.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Generic filename for dataset to read into memory like a 'csv' file. Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                files. Default is None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            verbose (int): Logging level. Default is 10.
        """
        super(MemoryGraphDataset, self).__init__()
        # For logging.
        self.logger = logging.getLogger("kgcnn.data." + dataset_name) if dataset_name is not None else module_logger
        self.logger.setLevel(verbose)
        # Dataset information on file.
        self.data_directory = data_directory
        self.file_name = file_name
        self.file_directory = file_directory
        self.dataset_name = dataset_name
        # Data Frame for information.
        self.data_frame = None
        self.data_keys = None
        self.data_unit = None

    @property
    def file_path(self):
        if self.data_directory is None:
            self.warning("Data directory is not set.")
            return None
        if not os.path.exists(os.path.realpath(self.data_directory)):
            self.error("Data directory does not exist.")
        if self.file_name is None:
            self.warning("Can not determine file path.")
            return None
        return os.path.join(self.data_directory, self.file_name)

    @property
    def file_directory_path(self):
        if self.data_directory is None:
            self.warning("Data directory is not set.")
            return None
        if not os.path.exists(self.data_directory):
            self.error("Data directory does not exist.")
        if self.file_directory is None:
            self.warning("Can not determine file directory.")
            return None
        return os.path.join(self.data_directory, self.file_directory)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def save(self, filepath: str = None):
        r"""Save all graph properties to as dictionary as pickled file. By default saves a file named
        :obj:`dataset_name.kgcnn.pickle` in :obj:`data_directory`.

        Args:
            filepath (str): Full path of output file. Default is None.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, self.dataset_name + ".kgcnn.pickle")
        self.info("Pickle dataset...")
        save_pickle_file([x.to_dict() for x in self._list], filepath)
        return self

    def load(self, filepath: str = None):
        r"""Load graph properties from a pickled file. By default loads a file named
        :obj:`dataset_name.kgcnn.pickle` in :obj:`data_directory`.

        Args:
            filepath (str): Full path of input file.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, self.dataset_name + ".kgcnn.pickle")
        self.info("Load pickled dataset...")
        in_list = load_pickle_file(filepath)
        self._list = [GraphDict(x) for x in in_list]
        return self

    def read_in_table_file(self, file_path: str = None, **kwargs):
        r"""Read a data frame in :obj:`data_frame` from file path. By default uses :obj:`file_name` and pandas.
        Checks for a '.csv' file and then for excel file endings. Meaning the file extension of file_path is ignored
        but must be any of the following '.csv', '.xls', '.xlsx', 'odt'.

        Args:
            file_path (str): File path to table file. Default is None.
            kwargs: Kwargs for pandas :obj:`read_csv` function.

        Returns:
            self
        """
        if file_path is None:
            file_path = os.path.join(self.data_directory, self.file_name)
        file_path_base = os.path.splitext(file_path)[0]
        # file_extension_given = os.path.splitext(file_path)[1]

        for file_extension in [".csv"]:
            if os.path.exists(file_path_base + file_extension):
                self.data_frame = pd.read_csv(file_path_base + file_extension, **kwargs)
                return self
        for file_extension in [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]:
            if os.path.exists(file_path_base + file_extension):
                self.data_frame = pd.read_excel(file_path_base + file_extension, **kwargs)
                return self

        self.warning("Unsupported data extension of '%s' for table file." % file_path)
        return self

    def assert_valid_model_input(self, hyper_input: list, raise_error_on_fail: bool = True):
        r"""Interface to hyperparameter. Check whether dataset has graph-properties (tensor format) requested
        by model input. The model input is set up by a list of layer configs for the keras :obj:`Input` layer.

        Example:
            [{"shape": [None, 8710], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],

        Args:
            hyper_input (list): List of properties that need to be available to a model for training.
            raise_error_on_fail (bool): Whether to raise an error if assertion failed.
        """
        dataset = self

        def message_error(msg):
            if raise_error_on_fail:
                raise ValueError(msg)
            else:
                dataset.error(msg)

        for x in hyper_input:
            if "name" not in x:
                message_error("Can not infer name from '%s' for model input." % x)
            data = [dataset[i].obtain_property(x["name"]) for i in range(len(dataset))]
            if any([y is None for y in data]):
                message_error("Property %s is not defined for all graphs in list. Please run clean()." % x["name"])
            # we also will check shape here but only with first element.
            if hasattr(data[0], "shape") and "shape" in x:
                shape_element = data[0].shape
                shape_input = x["shape"]
                if len(shape_input) != len(shape_element):
                    message_error(
                        "Mismatch in rank for model input {} vs. {}".format(shape_element, shape_input))
                for i, dim in enumerate(shape_input):
                    if dim is not None:
                        if shape_element[i] != dim:
                            message_error(
                                "Mismatch in shape for model input {} vs. {}".format(shape_element, shape_input))
            else:
                message_error("Can not check shape for '%s'." % x["name"])
        return

    def set_methods(self, method_list: list):
        for method_item in method_list:
            for method, kwargs in method_item.items():
                if hasattr(self, method):
                    getattr(self, method)(**kwargs)
                else:
                    self.error("Class does not have method '%s'." % method)


MemoryGeometricGraphDataset = MemoryGraphDataset
