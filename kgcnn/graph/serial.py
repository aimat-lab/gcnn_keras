import importlib
from typing import Union
from kgcnn.utils.serial import deserialize


def get_preprocessor(name: Union[str, dict], **kwargs):
    """Get a preprocessor.

    Args:
        name (str, dict): Serialization dictionary of class. This can also be a name of former graph functions for
            backward compatibility that now coincides with the processor's default name.
        kwargs: Kwargs for processor initialization, if :obj:`name` is string.

    Returns:
        GraphPreProcessorBase: Instance of graph preprocessor.
    """
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
        "principal_moments_of_inertia": "PrincipalMomentsOfInertia",
    }
    if isinstance(name, dict):
        return deserialize(name)
    # if given as string name. Lookup identifier.
    obj_class = getattr(importlib.import_module(str("kgcnn.graph.preprocessor")), str(preprocessor_identifier[name]))
    return obj_class(**kwargs)


def get_postprocessor(name: Union[str, dict], **kwargs):
    r"""Get a postprocessor.

    Args:
        name (str, dict): Serialization dictionary of class. This can also be a name of former graph functions for
            backward compatibility that now coincides with the processor's default name.
        kwargs: Kwargs for processor initialization, if :obj:`name` is string.

    Returns:
        GraphPostProcessorBase: Instance of graph postprocessor.
    """
    preprocessor_identifier = {
        "extensive_energy_force_scaler": "ExtensiveEnergyForceScalerPostprocessor",
    }
    if isinstance(name, dict):
        return deserialize(name)
    # if given as string. Lookup identifier.
    obj_class = getattr(importlib.import_module(str("kgcnn.graph.postprocessor")), str(preprocessor_identifier[name]))
    return obj_class(**kwargs)