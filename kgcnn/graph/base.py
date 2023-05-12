import numpy as np
import re
import logging
import networkx as nx
from collections.abc import MutableMapping
from kgcnn.graph.serial import get_preprocessor
from typing import Any, Union, List
from copy import deepcopy, copy

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


# Base classes are collections.UserDict or collections.abc.MutableMapping. However, this comes with more
# code to replicate dict-behaviour. Here GraphDict inherits from dict. Checking for numpy arrays can be disabled
# in class variable. Set item is completely free.
class GraphDict(dict):
    r"""Dictionary container to store graph information in tensor-form.

    The tensors must be stored as numpy arrays. The naming convention is not restricted.
    The class is supposed to be handled just as a python dictionary.

    In addition, :obj:`assign_property` and :obj:`obtain_property` handles `None` values and cast into tensor format,
    when assigning a named value.

    Graph operations that modify edges or sort indices can be applied via :obj:`apply_preprocessor` located in
    :obj:`kgcnn.graph.preprocessor`.

    .. code-block:: python

        import numpy as np
        from kgcnn.graph.base import GraphDict
        graph = GraphDict({"edge_indices": np.array([[1, 0], [0, 1]]), "edge_labels": np.array([[-1], [1]])})
        graph.set("graph_labels", [0])  # opposite is get()
        graph.set("edge_attributes", [[1.0], [2.0]])
        graph.apply_preprocessor("add_edge_self_loops")
        graph.apply_preprocessor("sort_edge_indices")
        print(graph)

    """

    _require_str_key = True
    _cast_to_array = True
    _tensor_class = np.ndarray
    _tensor_conversion = np.array
    _require_validate = True

    def __init__(self, *args, **kwargs):
        r"""Initialize a new :obj:`GraphDict` instance."""
        super(GraphDict, self).__init__(*args, **kwargs)
        if self._require_validate:
            self.validate()
        
    def update(self, *args, **kwargs):
        super(GraphDict, self).update(*args, **kwargs)
        if self._require_validate:
            self.validate()

    def assign_property(self, key: str, value):
        r"""Add a named property as key, value pair to self. If the value is `None`, nothing is done.
        Similar to assign-item default method :obj:`__setitem__`, but ignores `None` values and casts to tensor.

        Args:
            key (str): Name of the graph tensor to add to self.
            value: Array or tensor value to add. Can also be None.

        Returns:
            None.
        """
        if self._require_str_key:
            if not isinstance(key, str):
                raise ValueError("`GraphDict` requires string keys, but kot '%s'." % key)
        if value is not None:
            if self._cast_to_array:
                if not isinstance(value, self._tensor_class):
                    value = self._tensor_conversion(value)
            self[key] = value

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

    def validate(self):
        """Routine to check if items are set correctly, i.e. string key and np.ndarray values.
        Could be done manually."""
        for key, value in self.items():
            if self._require_str_key:
                if not isinstance(key, str):
                    raise ValueError("`GraphDict` requires string keys, but kot '%s'." % key)
            if self._cast_to_array:
                if not isinstance(value, self._tensor_class):
                    self[key] = self._tensor_conversion(value)

    # Alias of internal assign_property and obtain property.
    set = assign_property
    # get = obtain_property

    def copy(self):
        return GraphDict({key: copy(value) for key, value in self.items()})

    def to_dict(self) -> dict:
        """Returns a python-dictionary of self. Does not copy values.

        Returns:
            dict: Dictionary of graph tensor objects.
        """
        return {key: value for key, value in self.items()}

    def search_properties(self, keys: Union[str, list]) -> list:
        r"""Search for properties in self.

        This includes a list of possible names or a pattern-matching of a single string.

        Args:
            keys (str, list): Pattern matching string or list of strings to search for.

        Returns:
            list: List of names in self that match :obj:`keys`.
        """
        if keys is None:
            return []
        elif isinstance(keys, str):
            match_props = []
            for x in self:
                if re.match(keys, x):
                    if re.match(keys, x).group() == x:
                        match_props.append(x)
            return sorted(match_props)
        elif isinstance(keys, (list, tuple)):
            # No pattern matching for list input.
            return sorted([x for x in keys if x in self])
        return []

    # Old Alias
    find_graph_properties = search_properties

    def assert_has_valid_key(self, key: str, raise_error: bool = True) -> bool:
        """Assert the property is found in self.

        Args:
            key (str): Name of property that must be defined.
            raise_error (bool): Whether to raise error. Default is True.

        Returns:
            bool: Key is valid. Only if `raise_error` is False.
        """
        if key not in self or self.obtain_property(key) is None:
            if raise_error:
                raise AssertionError("`GraphDict` does not have '%s' key." % key)
            return False
        return True

    def has_valid_key(self, key: str) -> bool:
        """Check if the property is found in self and also is not Noe.

        Args:
            key (str): Name of property that must be defined.

        Returns:
            bool: Key is valid. Only if `raise_error` is False.
        """
        if key not in self or self.obtain_property(key) is None:
            return False
        return True

    def from_networkx(self, graph: nx.Graph,
                      node_number: str = "node_number",
                      edge_indices: str = "edge_indices",
                      node_attributes: Union[str, List[str]] = None,
                      edge_attributes: Union[str, List[str]] = None,
                      graph_attributes: Union[str, List[str]] = None,
                      node_labels: str = None,
                      reverse_edge_indices: bool = False):
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
            graph_attributes (str, list): Name of graph attributes to add from graph data. Can also be a list of names.
                Default is None.
            node_labels (str): Name of the labels of nodes to store former node IDs into. Default is None.
            reverse_edge_indices (bool): Whether to reverse edge indices for notation '(ij, i<-j)'. Default is False.

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
        node_attr_dict = {x: [None] * graph_size for x in node_attr}
        nodes_id = []
        for i, x in enumerate(graph_int.nodes.data()):
            nodes_id.append(x[0])
            for d in node_attr:
                if d not in x[1]:
                    raise KeyError("Node does not have property '%s'." % d)
                node_attr_dict[d][i] = x[1][d]

        # Loop over edges in graph.
        edge_id = []
        edges_attr = _attr_to_list(edge_attributes)
        edges_attr_dict = {x: [None] * graph_int.number_of_edges() for x in edges_attr}
        for i, x in enumerate(graph_int.edges.data()):
            if reverse_edge_indices:
                edge_id.append([x[1], x[0]])
            else:
                edge_id.append([x[0], x[1]])
            for d in edges_attr:
                if d not in x[2]:
                    raise KeyError("Edge does not have property '%s'." % d)
                edges_attr_dict[d][i] = x[2][d]

        graph_attr = _attr_to_list(graph_attributes)
        graph_attr_dict = {x: None for x in graph_attr}
        # We use original graph input for this. Does not need node relabeling.
        for _, x in enumerate(graph_attr):
            # This is a temporary solution until we find a better way to store graph-level information.
            if hasattr(graph, x):
                graph_attr_dict[x] = getattr(graph, x)

        # Storing graph tensors in self.
        self.assign_property(node_number, nodes_id)
        self.assign_property(edge_indices, edge_id)
        for key, value in node_attr_dict.items():
            self.assign_property(key, value)
        for key, value in edges_attr_dict.items():
            self.assign_property(key, value)
        for key, value in graph_attr_dict.items():
            self.assign_property(key, value)
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

    def apply_preprocessor(self, name, **kwargs):
        r"""Apply a preprocessor on self.

        Args:
            name: Name of a preprocessor that uses :obj:`kgcnn.graph.serial.get_preprocessor` for backward
                compatibility, or a proper serialization dictionary or class :obj:`GraphPreProcessorBase`.
            kwargs: Optional kwargs for preprocessor. Only used in connection with
                :obj:`kgcnn.graph.serial.get_preprocessor` if :obj:`name` is string.

        Returns:
            self
        """
        if isinstance(name, str):
            proc_graph = get_preprocessor(name, **kwargs)(self)
            # print(proc_graph)
            self.update(proc_graph)
            return self
        elif isinstance(name, GraphPreProcessorBase):
            proc_graph = name(self)
            self.update(proc_graph)
            return self
        raise ValueError("Unsupported preprocessor '%s'." % name)


# Deprecated alias for GraphDict.
GraphNumpyContainer = GraphDict


class GraphProcessorBase:
    r"""General base class for graph processors.

    A graph processor should be a subclass that implements a call methods and store all relevant information on how
    to change the graph properties and how they are supposed to be named in the :obj:`GraphDict` .

    .. code-block:: python

        from kgcnn.graph.base import GraphDict, GraphProcessorBase

        class NewProcessor(GraphProcessorBase):

            def __init__(self, name="new_processor", node_property_name="node_number"):
                self.node_property_name = node_property_name
                self.name = name

            def __call__(self, graph: GraphDict) -> GraphDict:
                # Do nothing but return the same property.
                return GraphDict({self.node_property_name: graph[self.node_property_name]})

            def get_config(self):
                return {"name": self.name, "node_property_name": self.node_property_name}

    The class provides utility functions to assign and obtain graph properties to or from a :obj:`GraphDict` by name.
    Since the namespace on how to name properties in a general :obj:`GraphDict` is not restricted, subclasses of
    :obj:`GraphProcessorBase` must set the property names in class construction and then can use the utility functions
    :obj:`_obtain_properties` and :obj:`_assign_properties`.

    Additionally, :obj:`_obtain_properties` and :obj:`_assign_properties` can contain a flexible list of properties
    or a search string that will return multiple properties. This is mainly required to extract a flexible list of
    properties, for example all edge-related properties via the search string '^edge_(?!indices$).*' that starts with
    'edge_*' but not 'edge_indices'. Note that you can always cast to a networkx graph, operate your own functions,
    then transform back into tensor form and wrap it as a :obj:`GraphProcessorBase` .
    """

    @staticmethod
    def _obtain_properties(graph: GraphDict, to_obtain: dict, to_search, to_silent) -> dict:
        r"""Extract a dictionary of named properties from a :obj:`GraphDict`.

        Args:
            graph (GraphDict): Dictionary with of graph properties.
            to_obtain (dict): Dictionary of keys that holds names or a list of names or a search string of graph
                properties to fetch. The output dictionary will then have the same keys but with arrays in place of the
                property name(s). The keys can be used as function arguments for some transform function.
            to_search (list): A list of strings that should be considered search strings.
            to_silent (list): A list of strings that suppress 'error not found' messages.

        Returns:
            dict: A dictionary of resolved graph properties in tensor form.
        """
        obtained_properties = {}
        for key, name in to_obtain.items():
            if isinstance(name, str):
                if name in to_search:
                    names = graph.search_properties(name)  # Will be sorted list and existing only
                    obtained_properties[key] = [graph.obtain_property(x) for x in names]
                else:
                    if not graph.has_valid_key(name) and key not in to_silent:
                        module_logger.warning("Missing '%s' in '%s'" % (name, type(graph).__name__))
                    obtained_properties[key] = graph.obtain_property(name)
            elif isinstance(name, (list, tuple)):
                prop_list = []
                for x in name:
                    if not graph.has_valid_key(x) and key not in to_silent:
                        module_logger.warning("Missing '%s' in '%s'" % (x, type(graph).__name__))
                    prop_list.append(graph.obtain_property(x))
                obtained_properties[key] = prop_list
            else:
                raise ValueError("Unsupported property identifier %s" % name)
        return obtained_properties

    @staticmethod
    def _assign_properties(graph: GraphDict, graph_properties: Union[list, np.ndarray],
                           to_assign, to_search, in_place=False) -> Union[dict, GraphDict]:
        r"""Assign a list of arrays to a :obj:`GraphDict` by name.

        Args:
            graph (GraphDict): Dictionary with of graph properties. That can be changed in place. Also required to
                resolve search strings.
            graph_properties (list, np.ndarray): Array or list of arrays which are new graph properties
            to_assign (str, list): Name-keys or list of name-keys for the :obj:`graph_properties`. If :obj:`to_assign`
                is list, the length must match :obj:`graph_properties` .
            to_search (list): A list of strings that should be considered search strings.
                Note that in this case there must be a list of arrays in place of a single array, since search
                strings will return a list of matching strings.
            in_place (bool): Whether to update :obj:`graph` argument.

        Returns:
            dict: Dictionary of arrays matched with name-keys.
        """
        out_graph = GraphDict()

        def _check_list_property(n, p):
            if not isinstance(p, (list, tuple)):
                module_logger.error("Wrong return type for '%s', which is not list." % n)
            if len(n) != len(p):
                module_logger.error("Wrong number of properties '%s' for '%s'." % (p, n))

        def _assign_single(name, single_graph_property):
            if isinstance(name, str):
                if name in to_search:
                    names = graph.search_properties(name)  # Will be sorted list and existing only
                    # Assume that names matches graph_properties
                    _check_list_property(names, single_graph_property)
                    for x, gp in zip(names, single_graph_property):
                        out_graph.assign_property(x, gp)
                else:
                    out_graph.assign_property(name, single_graph_property)
            elif isinstance(name, (list, tuple)):
                # For nested output
                _check_list_property(name, single_graph_property)
                for x, gp in zip(name, single_graph_property):
                    out_graph.assign_property(x, gp)
            else:
                module_logger.error("Wrong type of named property '%s'" % name)
            return

        # Process assignment here.
        if isinstance(to_assign, str):
            _assign_single(to_assign, graph_properties)
        elif isinstance(to_assign, (list, tuple)):
            _check_list_property(to_assign, graph_properties)
            for key, value in zip(to_assign, graph_properties):
                _assign_single(key, value)

        if in_place:
            graph.update(out_graph)
        return out_graph

    @staticmethod
    def has_special_characters(query, pat=re.compile("[@\\\\!#$%^&*()<>?/|}{~:]")):
        """Whether a query string has special characters."""
        return pat.search(query) is not None


class GraphPreProcessorBase(GraphProcessorBase):
    r"""Base wrapper for a graph preprocessor to use general transformation functions.

    This class inherits from :obj:`GraphProcessorBase` and makes use of :obj:`_obtain_properties` and
    :obj:`_assign_properties` in a predefined :obj:`__call__` method. The base class :obj:`__call__` method retrieves
    graph properties by name, passes them as kwargs to the :obj:`call` method, which must be implemented by subclasses,
    and assigns the (array-like) output to a :obj:`GraphDict` by name. To specify the kwargs for call and the names that
    are expected to be set in the graph dictionary, the class attributes :obj:`self._to_obtain` ,
    :obj:`self._to_assign` , :obj:`self._to_assign` , :obj:`self._call_kwargs` , :obj:`self._search` must be set
    appropriately in the constructor. The design of this class should be similar to :obj:`ks.layers.Layer` .
    Example implementation for subclass:

    .. code-block:: python

        from kgcnn.graph.base import GraphDict, GraphPreProcessorBase

        class NewProcessor(GraphPreProcessorBase):

            def __init__(self, name="new_processor", node_property_name="node_number", **kwargs)
                super().__init__(name=name, **kwargs):
                self._to_obtain = {"nodes": node_property_name}  # what 'call' needs as properties
                self._to_assign = ["new_nodes"]  # will be the output of call()
                self._call_kwargs = {"do_nothing": True}
                self._config_kwargs.update({"node_property_name": node_property_name})

            def call(self, nodes: np.ndarray, do_nothing: bool):
                # Do nothing but return the same property.
                return nodes

    For proper serialization you should update :obj:`self._config_kwargs` . For information on :obj:`self._search` and
    :obj:`self._silent` see docs of :obj:`GraphProcessorBase` .

    """

    def __init__(self, *, in_place: bool = False, name: str = None):
        r"""Initialize class.

        Args:
            in_place (bool): Whether to update graph dict. Default is False.
            name: Name of the preprocessor.
        """
        self._config_kwargs = {"in_place": in_place, "name": name}
        self._in_place = in_place
        # Below attributes must be set by subclass.
        self._to_obtain = {}
        self._to_assign = None
        self._call_kwargs = {}
        self._search = []
        self._silent = []

    def call(self, **kwargs):
        raise NotImplementedError("Must be implemented in sub-class.")

    def __call__(self, graph: GraphDict):
        r"""Processing the graph with function given in :obj:`call` .

        Args:
            graph (GraphDict): Graph dictionary to process.

        Returns:
            dict: Dictionary of new properties.
        """
        graph_properties = self._obtain_properties(graph, self._to_obtain, self._search, self._silent)
        # print(graph_properties)
        processed_properties = self.call(**graph_properties, **self._call_kwargs)
        out_graph = self._assign_properties(graph, processed_properties, self._to_assign, self._search)
        # print(out_graph)
        if self._in_place:
            graph.update(out_graph)
            return graph
        return out_graph

    def get_config(self):
        r"""Copy config from :obj:`self._config_kwargs` ."""
        config = deepcopy(self._config_kwargs)
        return config


class GraphPostProcessorBase(GraphProcessorBase):
    r"""Base wrapper for a graph postprocessor to use general transformation functions.

    This class inherits from :obj:`GraphProcessorBase` and makes use of :obj:`_obtain_properties` and
    :obj:`_assign_properties` in a predefined :obj:`__call__` method. The base class :obj:`__call__` method retrieves
    graph properties by name, passes them as kwargs to the :obj:`call` method, which must be implemented by subclasses,
    and assigns the (array-like) output to a :obj:`GraphDict` by name. To specify the kwargs for call and the names that
    are expected to be set in the graph dictionary, the class attributes :obj:`self._to_obtain` ,
    :obj:`self._to_assign` , :obj:`self._to_assign` , :obj:`self._call_kwargs` , :obj:`self._search` must be set
    appropriately in the constructor. The design of this class should be similar to :obj:`ks.layers.Layer` .
    Example implementation for subclass:

    .. code-block:: python

        from kgcnn.graph.base import GraphDict, GraphPostProcessorBase

        class NewProcessor(GraphPostProcessorBase):

            def __init__(self, name="new_processor", node_property_name="node_number", **kwargs)
                super().__init__(name=name, **kwargs):
                self._to_obtain = {"nodes": node_property_name}  # what 'call' needs as properties
                self._to_obtain_pre = {}
                self._to_assign = ["new_nodes"]  # will be the output of call()
                self._call_kwargs = {"do_nothing": True}
                self._config_kwargs.update({"node_property_name": node_property_name})

            def call(self, nodes: np.ndarray, do_nothing: bool):
                # Do nothing but return the same property.
                return nodes

    For proper serialization you should update :obj:`self._config_kwargs` . For information on :obj:`self._search` and
    :obj:`self._silent` see docs of :obj:`GraphProcessorBase` .

    Note that for :obj:`GraphPostProcessorBase` one can also set :obj:`self._to_obtain_pre` if information of the input
    graph or a previous graph is required, which is common for postprocessing.
    """

    def __init__(self, *, in_place: bool = False, name: str = None):
        r"""Initialize class.

        Args:
            in_place (bool): Whether to update graph dict. Default is False.
            name: Name of the preprocessor.
        """
        self._config_kwargs = {"in_place": in_place, "name": name}
        self._in_place = in_place
        # Below attributes must be set by subclass.
        self._to_obtain = {}
        self._to_obtain_pre = {}
        self._to_assign = None
        self._call_kwargs = {}
        self._search = []
        self._silent = []

    def call(self, **kwargs):
        raise NotImplementedError("Must be implemented in sub-class.")

    def __call__(self, graph: GraphDict, pre_graph: GraphDict = None):
        r"""Processing the graph with function given in :obj:`call` .

        Args:
            graph (GraphDict): Graph dictionary to process.
            pre_graph (GraphDict): Additional graph dictionary with properties to include in :obj:`call` .
                This is the main difference to preprocessor. Defaults to None.

        Returns:
            dict: Dictionary of new properties.
        """
        graph_properties_y = self._obtain_properties(graph, self._to_obtain, self._search, self._silent)

        if pre_graph is not None:
            graph_properties_x = self._obtain_properties(pre_graph, self._to_obtain_pre, self._search, self._silent)
            processed_properties = self.call(**graph_properties_y, **graph_properties_x, **self._call_kwargs)
        else:
            processed_properties = self.call(**graph_properties_y, **self._call_kwargs)

        out_graph = self._assign_properties(graph, processed_properties, self._to_assign, self._search)
        if self._in_place:
            graph.update(out_graph)
            return graph
        return out_graph

    def get_config(self):
        r"""Copy config from :obj:`self._config_kwargs` ."""
        config = deepcopy(self._config_kwargs)
        return config
