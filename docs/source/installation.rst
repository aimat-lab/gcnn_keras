.. _installation:
   :maxdepth: 3

Installation
============

Clone repository https://github.com/aimat-lab/gcnn_keras and install with editable mode::

   pip install -e ./gcnn_keras

or latest release via `Python Package Index <https://pypi.org/>`_ ::

   pip install kgcnn

Standard python package requirements are placed in the ``setup.py`` and are installed automatically (`kgcnn <https://github.com/aimat-lab/gcnn_keras>`_ >=2.2).
Packages which must be installed manually for full functionality are listed below.
For example to convert molecular file formats `OpenBabel <http://openbabel.org/wiki/Main_Page>`_ can complement `RDkit <https://www.rdkit.org/docs/api-docs.html>`_ which is installed via ``pip``.

* `OpenBabel <http://openbabel.org/wiki/Main_Page>`_ >=3.0.1

To have proper GPU support, make sure that the installed ``tensorflow`` version matches your system requirements.
Moreover, installed `GPU drivers <https://www.nvidia.com/download/index.aspx?lang=en-us>`_ and `CUDA <https://developer.nvidia.com/cuda-toolkit-archive>`_  and `cuDNN <https://developer.nvidia.com/cudnn>`_ versions must match.
A list of verified version combinations can be found here: https://www.tensorflow.org/install/source#gpu .