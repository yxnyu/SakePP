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
```

#### Option 1: Create conda environment and install dependencies
./install.sh  

#### Option 2: PIP Install package
```bash
pip install -r requirements.txt
```
or, alternatively,
```bash
pip install -e .
```


### Launch training
To re-construct by using default settings,
at the very first please make sure ``` .h5```dataset and ``` .pt```checkpoint already exist in the given directory:
```
SakePP/
  |- weights/
    |- 2024pdb.h5
    |- process_pdb_0311.pt
```
You can download these files from Google Drive: 
[2024pdb.h5](https://drive.google.com/file/d/xxxxxx.h5)  [process_pdb_0311.pt](https://drive.google.com/file/d/xxxxxx.pt) 

Then, run the main script:
```bash
python main.py
```
To customise your training settings, i.e. dataset or pretrained weight, or using a different path importing your dataset,
please edit ``` ./config/config.yaml```, and put your files in the correct directory correspondingly.

To customise the model, i.e.  hyper-parameters of model,
please edit ``` ./config/models/models.yaml```.




## Development Roadmap

Because the current workflow uses HDF5-based batch I/O for `.pt` tensors in both training and testing, we are actively 
developing the following features to enhance usability and scalability:

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


