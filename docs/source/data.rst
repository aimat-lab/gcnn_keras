Data
----

Graph dictionary
----------------

Graphs are represented by a dictionary ``GraphDict`` of (numpy) arrays
which behaves like a python dict. There are graph pre- and
postprocessors in ``kgcnn.graph`` which take specific properties by name
and apply a processing function or transformation.

.. code:: ipython3

    from kgcnn.data.base import GraphDict
    # Single graph.
    graph = GraphDict({"edge_indices": [[1, 0], [0, 1]], "node_label": [[0], [1]]})
    graph.set("graph_labels", [0])  # use set(), get() to assign (tensor) properties.
    graph.set("edge_attributes", [[1.0], [2.0]])
    graph.to_networkx()
    # Modify with e.g. preprocessor.
    from kgcnn.graph.preprocessor import SortEdgeIndices
    SortEdgeIndices(edge_indices="edge_indices", edge_attributes="^edge_(?!indices$).*", in_place=True)(graph);

Datasets
--------

Loading Options
---------------

