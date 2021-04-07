import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from kgcnn.literature.GNNExplain import GNNExplainer,GNNExplainerOptimizer,GNNInterface

from kgcnn.utils.adj import precompute_adjacency_scaled
from kgcnn.literature.GCN import make_gcn
from kgcnn.utils.learning import lr_lin_reduction

from kgcnn.data.cora.cora_lu import cora_graph

nodes, edge_index, labels, class_label_mapping = cora_graph()
nodes = nodes [:,1:] # Remove IDs