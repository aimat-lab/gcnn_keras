from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="kgcnn",
    version="0.1.1",
    author="Patrick Reiser",
    author_email="patrick.reiser@kit.edu",
    description="General Base Layers for Graph Convolutions with tensorflow.keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aimat-lab/gcnn_keras",
    install_requires=['numpy', "scikit-learn", "pandas", "scipy", "requests"],
    extras_require={
        "tf": ["tensorflow>=2.2.0"],
        "tf_gpu": ["tensorflow-gpu>=2.2.0"],
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={"kgcnn": ["*.json", "*.yaml"]},
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
