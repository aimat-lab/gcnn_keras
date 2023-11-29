.. _installation:
   :maxdepth: 3

Installation
============

Clone repository https://github.com/aimat-lab/gcnn_keras and install with editable mode
or latest release via `Python Package Index <https://pypi.org/>`_ ::

   pip install kgcnn


Standard python package requirements are installed automatically.
However, you must make sure to install the GPU/TPU acceleration for the backend of your choice.
For example, to have proper GPU support, make sure that the installed ``tensorflow`` version matches your system requirements.
Moreover, installed `GPU drivers <https://www.nvidia.com/download/index.aspx?lang=en-us>`_ and `CUDA <https://developer.nvidia.com/cuda-toolkit-archive>`_  and `cuDNN <https://developer.nvidia.com/cudnn>`_ versions must match.
A list of verified version combinations for tensorflow can be found here: https://www.tensorflow.org/install/source#gpu .
