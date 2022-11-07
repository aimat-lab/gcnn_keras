import importlib


def get_preprocessor(name, **kwargs):
    """Get Preprocessor by name, for compatibility to old class methods."""
    preprocessor_identifier = {
        "make_undirected_edges": "MakeUndirectedEdges",
        "add_edge_self_loops": "AddEdgeSelfLoops",
        "sort_edge_indices": "SortEdgeIndices",
        "set_edge_indices_reverse": "SetEdgeIndicesReverse",
        "pad_property": "PadProperty",
        "set_edge_weights_uniform": "SetEdgeWeightsUniform",
        "normalize_edge_weights_sym": "NormalizeEdgeWeightsSymmetric",
        "set_range_from_edges": "SetRangeFromEdges",
        "set_range": "SetRange",
        "set_angle": "SetAngle",
        "set_range_periodic": "SetRangePeriodic",
        "expand_distance_gaussian_basis": "ExpandDistanceGaussianBasis",
        "atomic_charge_representation": "AtomicChargesRepresentation",
    }
    obj_class = getattr(importlib.import_module(str("kgcnn.graph.preprocessor")), str(preprocessor_identifier[name]))
    return obj_class(**kwargs)
