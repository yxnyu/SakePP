from setuptools import setup, find_packages
import pathlib
import re


def get_version():
    init_file = pathlib.Path(__file__).parent / "sakepp" / "__init__.py"
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read_text(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError(
        "Unable to find version string.")


setup(
    name="SakePP",
    version=get_version(),
    author="Yuzhi Xu, Xinxin Liu",
    author_email="yuzhixu@nyu.edu",
    description="SakePP Protein-Protein Interaction with a Spatial Attention Kinetic Graph Neural Network",

    url="https://github.com/yxnyu/SakePP",

    packages=find_packages(include=["sakepp", "sakepp.*"]),
    package_data={
        "sakepp": [
            "config/*.json",
            "data/sample/*.h5",
            "models/pretrained/*.pt"
        ],
    },

    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "dgl>=1.1.0",
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "scikit-learn>=1.0.0",
        "h5py>=3.1.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "scikit-learn",
        "hydra-core",
        "pandas"
    ],
    

    entry_points={
        "console_scripts": [
            "sake-train=sakepp.cli.train:main",
            "sake-predict=sakepp.cli.predict:main"
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],

    license="MIT",
    keywords="protein gnn deep-learning",
    zip_safe=False
)