import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler
from kgcnn.data.datasets.lipop import LipopDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelection
from kgcnn.training.graph import train_graph_regression_supervised

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Lipop dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="AttentiveFP")  # AttentiveFP
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.",
                    default="hyper/hyper_lipop.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model
model_name = args["model"]
model_selection = ModelSelection()
make_model = model_selection.make_model(model_name)

# Hyper-parameter.
hyper_selection = HyperSelection(args["hyper"], model_name=model_name, dataset_name="Lipop")
hyper = hyper_selection.hyper()

# Loading Lipop Dataset
hyper_data = hyper['data']
dataset = LipopDataset().set_attributes()
if "set_range" in hyper_data:
    dataset.set_range(**hyper_data["set_range"])
if "set_edge_indices_reverse" in hyper_data:
    dataset.set_edge_indices_reverse()
dataset_name = dataset.dataset_name
data_unit = "logD at pH 7.4"
data_length = dataset.length

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
data_loader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = np.array(dataset.graph_labels)

# Define Scaler for targets.
scaler = StandardScaler(with_std=True, with_mean=True, copy=True)

# Use a generic training function for graph regression.
train_graph_regression_supervised(data_loader, labels,
                                  make_model=make_model,
                                  hyper_selection=hyper_selection,
                                  scaler=scaler)