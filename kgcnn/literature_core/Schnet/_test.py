import numpy as np
from kgcnn.data.datasets.ESOLDataset import ESOLDataset

data = ESOLDataset()
data.map_list("set_range", max_distance=4)

from kgcnn.literature.Schnet import make_model as model1
conf_1 = {
    "name": "Schnet",
    "inputs": [
        {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
        {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
    ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "output_embedding": "graph",
    'output_mlp': {"use_bias": [True, True], "units": [64, 1],
                   "activation": ['kgcnn>shifted_softplus', "linear"]},
    'last_mlp': {"use_bias": [True, True], "units": [128, 64],
                 "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus']},
    "interaction_args": {
        "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
    },
    "node_pooling_args": {"pooling_method": "sum"},
    "depth": 0,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10
}
m1 = model1(**conf_1)

from kgcnn.literature_core.Schnet import make_model as model2
conf_2 = {
    "name": "Schnet",
    "inputs": [
        {"shape": [None], "name": "node_number", "dtype": "int32"},
        {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
        {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
        {"shape": (), "name": "graph_size", "dtype": "int64"},
        {"shape": (), "name": "edge_count", "dtype": "int64"}
    ],
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "output_embedding": "graph",
    'output_mlp': {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["kgcnn>shifted_softplus", "linear"]},
    'last_mlp': {"use_bias": [True, True], "units": [128, 64],
                 "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},
    "interaction_args": {
        "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus",
        "cfconv_pool": "scatter_sum"
    },
    "node_pooling_args": {"pooling_method": "scatter_sum"},
    "depth": 0,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4}, "verbose": 10
}
m2 = model2(**conf_2)
# m2.set_weights(m1.get_weights())

print(np.mean(np.abs(m2.predict(data.tensor(conf_2["inputs"])) - m1.predict(data.tensor(conf_1["inputs"])))))
m2.set_weights(m1.get_weights())
print(np.mean(np.abs(m2.predict(data.tensor(conf_2["inputs"])) - m1.predict(data.tensor(conf_1["inputs"])))))