import tensorflow as tf
import numpy as np
import os

from kgcnn.utils import learning
from kgcnn.utils.loss import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from kgcnn.data.datasets.LocalDataset import LocalDataset
from kgcnn.io.loader import NumpyTensorList
from kgcnn.utils.data import save_json_file, load_json_file

# Loading Local Dataset
dataset = LocalDataset(dataset_name='Test', local_full_path='/Users/tgg/Downloads/exts-ml/Denis_IngKnowledge/test.csv',columnsNames = 'Values1',reload=True).set_attributes()
print(dataset)
dir(dataset)