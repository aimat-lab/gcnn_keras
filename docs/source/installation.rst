.. _installation:
   :maxdepth: 3

Installation
============

Clone repository https://github.com/aimat-lab/gcnn_keras and install with editable mode::

   pip install -e ./gcnn_keras

or latest release via Python Package Index::

   pip install kgcnn

For kgcnn, usually the latest version of tensorflow is required, but is listed as extra requirements in the ``setup.py`` for simplicity.
Additional python packages are placed in the ``setup.py`` requirements and are installed automatically.
Packages which must be installed manually for full functionality:

* tensorflow>=2.4.1
* rdkit>=2020.03.4
* openbabel>=3.0.1
* pymatgen>=??.??.??
