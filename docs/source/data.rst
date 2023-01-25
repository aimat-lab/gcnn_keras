Data
====

There are some data classes in ``kgcnn`` that can help to store and load
graph data. In principle a graph is a collection of the follwing objects
in tensor-form:

-  ``nodes_attributes``: Node features of shape ``(N, F)`` where N is
   the number of nodes and F is the node feature dimension.
-  edge_indices:
-  edges_attributes:

Graph dictionary
----------------

Graphs are represented by a dictionary ``GraphDict`` of (numpy) arrays
which behaves like a python dict.

.. code:: ipython3

    import numpy as np
    from kgcnn.data.base import GraphDict
    # Single graph.
    graph = GraphDict({"edge_indices": np.array([[1, 0], [0, 1]]), "node_label":  np.array([[0], [1]])})
    graph.set("graph_labels",  np.array([0]))
    graph.set("edge_attributes", np.array([[1.0], [2.0]]));
    print({x: v.shape for x,v in graph.items()})


.. parsed-literal::

    {'edge_indices': (2, 2), 'node_label': (2, 1), 'graph_labels': (1,), 'edge_attributes': (2, 1)}
    

The class ``GraphDict`` can be converted to for example a strict Graph
representation of ``networkx`` which keeps track of node and edge
changes.

.. code:: ipython3

    nx_graph = graph.to_networkx()

There are graph pre- and postprocessors in ``kgcnn.graph`` which take
specific properties by name and apply a processing function or
transformation.

.. code:: ipython3

    from kgcnn.graph.preprocessor import SortEdgeIndices
    
    SortEdgeIndices(edge_indices="edge_indices", edge_attributes="^edge_(?!indices$).*", in_place=True)(graph);
    

Datasets
--------

Loading Options
===============

