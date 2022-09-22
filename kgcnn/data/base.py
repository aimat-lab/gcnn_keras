import logging
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from typing import Union, List
from collections.abc import MutableMapping
from kgcnn.data.utils import save_pickle_file, load_pickle_file, ragged_tensor_from_nested_numpy
from kgcnn.graph.base import GraphNumpyContainer, GraphDict

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class MemoryGraphList(MutableMapping):
    r"""Class to store a list of graph dictionaries in memory.

    Contains a python list as property :obj:`_list`. The graph properties are defined by tensor-like (numpy) arrays
    for indices, attributes, labels, symbol etc. in :obj:`GraphDict`, which are the items of the list.
    Access to items via `[]` indexing operator.

    A python list of a single named property can be obtained from each :obj:`GraphDict` in :obj:`MemoryGraphList` via
    :obj:`get` or assigned from a python list via :obj:`set` methods.

    The :obj:`MemoryGraphList` further provides simple map-functionality :obj:`map_list` to apply methods for
    each :obj:`GraphDict`, and to cast properties to tensor with :obj:`tensor`.

    Cleaning the list for missing properties or empty graphs is done with :obj:`clean`.

    .. code-block:: python

        import numpy as np
        from kgcnn.data.base import MemoryGraphList

        data = MemoryGraphList()
        data.empty(1)
        data.set("edge_indices", [np.array([[0, 1], [1, 0]])])
        data.set("node_labels", [np.array([[0], [1]])])
        print(data.get("edge_indices"))
        data.set("node_coordinates", [np.array([[1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])])
        data.map_list("set_range", max_distance=1.5, max_neighbours=10, self_loops=False)
        data.clean("range_indices")  # Returns cleaned graph indices
        print(len(data))
        print(data[0])
    """

    def __init__(self, input_list: list = None):
        r"""Initialize an empty :obj:`MemoryGraphList` instance.

        Args:
            input_list (list, MemoryGraphList): A list or :obj:`MemoryGraphList` of :obj:`GraphDict` items.
        """
        self._list = []
        self.logger = module_logger
        if input_list is None:
            input_list = []
        if isinstance(input_list, list):
            self._list = [GraphDict(x) for x in input_list]
        if isinstance(input_list, MemoryGraphList):
            self._list = [GraphDict(x) for x in input_list._list]

    def assign_property(self, key: str, value: list):
        """Assign a list of numpy arrays of a property to :obj:`GraphDict`s in this list.

        Args:
            key (str): Name of the property.
            value (list): List of numpy arrays for property `key`.

        Returns:
            self
        """
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

    def obtain_property(self, key: str) -> Union[list, None]:
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
        raise TypeError("Unsupported type for `MemoryGraphList` items.")

    def __setitem__(self, key, value):
        if not isinstance(value, GraphDict):
            raise TypeError("Require a GraphDict as list item.")
        self._list[key] = value

    def __delitem__(self, key):
        value = self._list.__delitem__(key)
        return value

    def __iter__(self):
        return iter(self._list)

    def _set_internal_list(self, value: list):
        if not isinstance(value, list):
            raise TypeError("Must set list for `MemoryGraphList` internal assignment.")
        self._list = value
        return self

    def clear(self):
        """Clear internal list.

        Returns:
            None
        """
        self._list.clear()

    def empty(self, length: int):
        """Create an empty list in place. Overwrites existing list.

        Args:
            length (int): Length of the empty list.

        Returns:
            self
        """
        if length is None:
            return self
        if length < 0:
            raise ValueError("Length of empty list must be >=0.")
        self._list = [GraphDict() for _ in range(length)]
        return self

    @property
    def length(self):
        """Length of list."""
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

    def tensor(self, items: Union[list, dict], make_copy=True):
        r"""Make tensor objects from multiple graph properties in list.

        It is recommended to run :obj:`clean` beforehand.

        Args:
            items (list): List of dictionaries that specify graph properties in list via 'name' key.
                The dict-items match the tensor input for :obj:`tf.keras.layers.Input` layers.
                Required dict-keys should be 'name' and 'ragged'.
                Optionally shape information can be included via 'shape'.
                E.g.: `[{'name': 'edge_indices', 'ragged': True}, {...}, ...]`.
            make_copy (bool): Whether to copy the data. Default is True.

        Returns:
            list: List of Tensors.
        """
        if isinstance(items, dict):
            return self._to_tensor(items, make_copy=make_copy)
        elif isinstance(items, (tuple, list)):
            return [self._to_tensor(x, make_copy=make_copy) for x in items]
        else:
            raise TypeError("Wrong type, expected e.g. [{'name': 'edge_indices', 'ragged': True}, {...}, ...]")

    def map_list(self, method: str, **kwargs):
        r"""Map a method over this list and apply on each :obj:`GraphDict`.

        .. code-block:: python

            for i, x in enumerate(self):
                getattr(x, method)(**kwargs)

        Args:
            method (str): Name of the :obj:`GraphDict` method.
            kwargs: Kwargs for `method`.

        Returns:
            self
        """
        for i, x in enumerate(self._list):
            # Can add progress info here.
            getattr(x, method)(**kwargs)
        return self

    def clean(self, inputs: Union[list, str]):
        r"""Given a list of property names, this method removes all elements from the internal list of
        `GraphDict` items, which do not define at least one of those properties. Meaning, only those graphs remain in
        the list which definitely define all properties specified by :obj:`inputs`.

        Args:
            inputs (list): A list of strings, where each string is supposed to be a property name, which the graphs
                in this list may possess. Within :obj:`kgcnn`, this can be simpy the 'input' category in model
                configuration. In this case, a list of dicts that specify the name of the property with 'name' key.

        Returns:
            invalid_graphs (np.ndarray): A list of graph indices that do not have the required properties and which
                have been removed.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        invalid_graphs = []
        for item in inputs:
            # If this is a list of dict, which are the config for ks.layers.Input(), we pick 'name'.
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
                        self.logger.info("Property %s is an empty list for graph %s." % (item_name, i))
                        invalid_graphs.append(i)
        invalid_graphs = np.unique(np.array(invalid_graphs, dtype="int"))
        invalid_graphs = np.flip(invalid_graphs)  # Need descending order for pop()
        if len(invalid_graphs) > 0:
            self.logger.warning("Found invalid graphs for properties. Removing graphs %s." % invalid_graphs)
        else:
            self.logger.info("No invalid graphs for assigned properties found.")
        # Remove from the end via pop().
        for i in invalid_graphs:
            self._list.pop(int(i))
        return invalid_graphs

    # Alias of internal assign and obtain property.
    set = assign_property
    get = obtain_property


class MemoryGraphDataset(MemoryGraphList):
    r"""Dataset class for lists of graph tensor dictionaries stored on file and fit into memory.

    This class inherits from :obj:`MemoryGraphList` and can be used (after loading and setup) as such.
    It has further information about a location on disk, i.e. a file directory and a file
    name as well as a name of the dataset.

    .. code-block:: python

        from kgcnn.data.base import MemoryGraphDataset
        dataset = MemoryGraphDataset(data_directory="", dataset_name="Example")
        # Methods of MemoryGraphList
        dataset.set("edge_indices", [np.array([[1, 0], [0, 1]])])
        dataset.set("edge_labels", [np.array([[0], [1]])])
        dataset.save()

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
        """Returns path information of `file_directory_`."""
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
        r"""Save all graph properties to python dictionary as pickled file. By default, saves a file named
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
        r"""Load graph properties from a pickled file. By default, loads a file named
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
        r"""Read a data frame in :obj:`data_frame` from file path. By default, uses :obj:`file_name` and pandas.
        Checks for a '.csv' file and then for Excel file endings. Meaning the file extension of file_path is ignored
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

        .. code-block:: python

            hyper_input = [
                {"shape": [None, 8710], "name": "node_attributes", "dtype": "float32", "ragged": True},
                {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
            ]

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

        def message_warning(msg):
            dataset.warning(msg)

        for x in hyper_input:
            if "name" not in x:
                message_error("Can not infer name from '%s' for model input." % x)
            data = [dataset[i].obtain_property(x["name"]) for i in range(len(dataset))]
            prop_in_data = [y is None for y in data]
            if all(prop_in_data):
                message_error("Property %s is not defined for any graph in list. Please check property." % x["name"])
            if any(prop_in_data):
                message_warning("Property %s is not defined for all graphs in list. Please run clean()." % x["name"])

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

    def set_methods(self, method_list: List[dict]) -> None:
        r"""Apply a list of serialized class-methods on the dataset.

        This can extend the config-serialization scheme in :obj:`kgcnn.utils.serial`.

        .. code-block:: python

            for method_item in method_list:
                for method, kwargs in method_item.items():
                    if hasattr(self, method):
                        getattr(self, method)(**kwargs)

        Args:
            method_list (list): A list of dictionaries that specify class methods. The `dict` key denotes the method
                and the value must contain `kwargs` for the method

        Returns:
            None.
        """
        for method_item in method_list:
            for method, kwargs in method_item.items():
                if hasattr(self, method):
                    getattr(self, method)(**kwargs)
                else:
                    self.error("Class does not have method '%s'." % method)


MemoryGeometricGraphDataset = MemoryGraphDataset
