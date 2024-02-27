# Main package version.
__kgcnn_version__ = "4.0.2"

# Global definition of index order and axis.
__indices_axis__ = 0
__index_receive__ = 0
__index_send__ = 1

# Behaviour for backend functions.
__safe_scatter_max_min_to_zero__ = True

# Geometry
__geom_euclidean_norm_add_eps__ = True  # Set to false for exact sqrt computation for geometric layers.
__geom_euclidean_norm_no_nan__ = True  # Only used for inverse norm.
