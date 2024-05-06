from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name="kgcnn",
    version="4.0.2",  # If version is updated, change version in `kgcnn.__init__` too. (and update changelog)
    author="Patrick Reiser",
    author_email="patrick.reiser@kit.edu",
    description="General Base Layers for Graph Convolutions with Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aimat-lab/gcnn_keras",
    install_requires=[
        # "dm-tree",
        "keras>=3.0.2",
        # Backends
        # "tensorflow-cpu>=2.16.1",
        # "torch>=2.1.0",
        # "torchvision>=0.16.0",
        # "jax[cpu]",
        # "torchrec",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.3",
        "pandas>=1.5.2",
        "scipy>=1.9.3",
        "matplotlib>=3.6.0",
        "rdkit>=2022.9.2",
        "pymatgen>=2022.11.7",
        "keras-tuner>=1.1.3",
        "requests>=2.28.1",
        "networkx>=2.8.8",
        "sympy>=1.11.1",
        "pyyaml>=6.0",
        "ase>=3.22.1",
        "click>=7.1.2",
        "brotli>=1.0.9",
        "h5py>=3.9.0",
        # PyXtal could be made optional.
        "pyxtal>=0.6.4"
    ],
    extras_require={
        "openbabel": ["openbabel"],
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={"kgcnn": ["*.json", "*.yaml", "*.csv", "*.md"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["materials", "science", "machine", "learning", "deep", "graph", "networks", "neural"]
)
