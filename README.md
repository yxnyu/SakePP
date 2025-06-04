SAKE-PP: Spatial Attention Kinetic GNN for Protein–Protein Interactions
==============================================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C.svg)](https://pytorch.org/)


## Overview

**SAKE-PP** is a spatially equivariant graph neural network for scoring protein–protein interaction (PPI) decoys via interface RMSD regression. 
The model combines Laplacian eigenvector-based orientation, physics-inspired attention, and geometric message passing to capture both local and 
global features of protein interfaces. It supports training and evaluation workflows with batch I/O using `.pt` tensors.

## Quick Start

### Installation

#### Clone the repository
```bash
git clone https://github.com/yxnyu/SakePP.git
cd SakePP
````

#### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```
or
```bash
.\venv\Scripts\activate  # Windows
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

### Usage

#### Train SAKE-PP on a custom dataset
```bash
```

#### Evaluate a pre-trained model on decoy inputs
```bash
```

#### Convert raw PDB data to graph tensor
```bash
```

## Development Roadmap

Because the current workflow uses HDF5-based batch I/O for `.pt` tensors in both training and testing, we are actively developing the following features to enhance usability and scalability:

* Native `.pdb` and `.mmCIF` support for end-to-end preprocessing
* Streaming data loaders for large-scale docking decoys
* Integration with AlphaFold3, ZDOCK, and other docking pipelines
* Visualization tools for inspecting spatial attention maps

## Citation

If you use SAKE-PP in your research or development, please cite:

```bibtex
@article{sakepp2024,
  title={Quantifying Protein-Protein Interaction with a Spatial Attention Kinetic Graph Neural Network},
  author={Yuzhi Xu et al.},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


