import numpy as np
from kgcnn.graph.adapter import GraphMethodsAdapter
import re
import networkx as nx


class GraphDict(dict, GraphMethodsAdapter):
    r"""Dictionary container to store graph information in tensor-form. At the moment only numpy-arrays are supported.
    The naming convention is not restricted. The class is supposed to be handled just as a python dictionary.
    In addition, :obj:`assign_property` and :obj:`obtain_property` handles `None` values and cast into tensor format,
    when assigning a named value.

    Graph operations that modify edges or sort indices are methods of this class supported by
    :obj:`kgcnn.graph.adapter.GraphMethodsAdapter`.
    Note that the graph-tensors name must follow a standard-convention or be provided to member functions
    (see documentation of :obj:`kgcnn.graph.adapter.GraphMethodsAdapter`).

    .. code-block:: python

        import numpy as np
        from kgcnn.graph.base import GraphDict
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

    # Alias of internal assign and obtain property.
    set = assign_property
    # get = obtain_property


GraphNumpyContainer = GraphDict
