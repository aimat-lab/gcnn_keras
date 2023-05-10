"""
Using `coGN and coNGN <https://arxiv.org/abs/2302.14102>`__ models in KGCNN.
============================================================================

There are multiple preconfigured configurations for `coGN
model <https://arxiv.org/abs/2302.14102>`__ for different input
representations. The input representations are different for, whether

-  the graph represents periodic/symmetric crystal graphs or
   non-periodic molecular graphs
-  the model must be differentiable with respect to node coordinates to
   calculate forces based on predicted energies.

Import all functions and configurations with:

.. code:: python

   from kgcnn.literature.coGN import (make_model, make_force_model,
       model_default, crystal_asymmetric_unit_graphs, molecular_graphs, crystal_unit_graphs,
       crystal_unit_graphs_coord_input, molecular_graphs_coord_input, model_default_nested)

Crystals
--------

-  The default coGN model for crystals (takes asymmetric unit graphs as
   input)

   .. code:: python

      model = make_model(**model_default)
      # model.inputs

   This is equivalent to:

   .. code:: python

      model = make_model(**crystal_asymmetric_unit_graphs)

   Factoring out symmetries via the asymmetric unit graph representation
   may accelerate training, since graphs are smaller.

-  For unit cells crystal graph representations use:

   .. code:: python

      model = make_model(**crystal_unit_graphs)
      # model.inputs

   Precomputing offsets between atoms in a preporcessing step may
   accelerate training.

-  For unit cell crystal graph representations and force predictions
   use:

   .. code:: python

      model = make_model(**crystal_unit_graphs_coord_input)
      force_model = make_force_model(model) # predicts energies and forces
      # model.inputs

Molecules
---------

-  For simple energy predictions, based on precomputed offsets between
   atoms use:

   .. code:: python

      model = make_model(**molecular_graphs)
      # model.inputs

   Precomputing offsets between atoms in a preporcessing step may
   accelerate training.

-  For energy and force predictions based on atom coordinates use:

   .. code:: python

      model = make_model(**molecular_graphs_coord_input)
      force_model = make_force_model(model)
      # model.inputs
"""

from ._make import make_model, make_force_model
from ._coGN_config import (model_default, crystal_asymmetric_unit_graphs, molecular_graphs, crystal_unit_graphs,
                           crystal_unit_graphs_coord_input, molecular_graphs_coord_input)
from ._coNGN_config import model_default_nested


__all__ = [
    "make_model",
    "make_force_model",
    "model_default",
    "crystal_asymmetric_unit_graphs",
    "crystal_unit_graphs",
    "crystal_unit_graphs_coord_input",
    "molecular_graphs",
    "molecular_graphs_coord_input",
    "model_default_nested"
]
