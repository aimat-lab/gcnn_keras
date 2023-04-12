import os
import numpy as np
from kgcnn.data.datasets.MatProjectJdft2dDataset import MatProjectJdft2dDataset
from kgcnn.literature.coGN import make_model, crystal_unit_graphs_coord_input


data = MatProjectJdft2dDataset()
data.map_list("set_range_periodic")
print(data[0].keys())

x_input = data[:10].tensor({
    "cell_translation": {"shape": (None, 3), "dtype": "float32", "name": "range_image", "ragged": True},
    "atomic_number": {"shape": (None,), "name": "node_number", "dtype": "int32", "ragged": True},
    "frac_coords": {"shape": (None,3), "dtype": "float32", "name": "node_frac_coordinates", "ragged": True},
    "edge_indices": {"shape": (None, 2), "name": "range_indices", "dtype": "int32", "ragged": True},
    "lattice_matrix": {"shape": (3,3), "dtype": "float32", "name": "graph_lattice"},
})

model = make_model(**crystal_unit_graphs_coord_input)
if not os.path.exists("weights.npy"):
    np.save("weights.npy", model.get_weights())
else:
    model.set_weights(np.load("weights.npy", allow_pickle=True))
out = model.predict(x_input)