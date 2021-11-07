import numpy as np
import argparse

from kgcnn.data.datasets.mutag import MUTAGDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.models import ModelSelection
from kgcnn.hyper.selection import HyperSelection
from kgcnn.training.graph import train_graph_classification_supervised

# Input arguments from command line.
# A hyper-parameter file can be specified to be loaded containing a python dict for hyper.
parser = argparse.ArgumentParser(description='Train a graph network on Mutagenicity dataset.')
parser.add_argument("--model", required=False, help="Graph model to train.", default="INorp")  # INorp
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config.", default="hyper/hyper_mutag.py")
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Model identification
model_name = args["model"]
ms = ModelSelection()
make_model = ms.make_model(model_name)

# Find hyper-parameter.
hyper_selection = HyperSelection(args["hyper"], model_name=model_name, dataset_name="MUTAG")
hyper = hyper_selection.get_hyper()

# Loading MUTAG Dataset
hyper_data = hyper['data']
dataset = MUTAGDataset()
dataset_name = dataset.dataset_name
data_length = dataset.length
if "set_edge_indices_reverse" in hyper_data:
    dataset.set_edge_indices_reverse()

# Using NumpyTensorList() to make tf.Tensor objects from a list of arrays.
dataloader = NumpyTensorList(*[getattr(dataset, x['name']) for x in hyper['model']['inputs']])
labels = np.expand_dims(dataset.graph_labels, axis=-1)

# Use a generic training function for graph classification.
train_graph_classification_supervised(dataloader, labels,
                                      make_model=make_model,
                                      hyper_selection=hyper_selection)
