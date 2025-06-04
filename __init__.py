# SakePP: Protein-Protein Interaction with a Spatial Attention Kinetic Graph Neural Network
# Author: Yuzhi Xu, Xinxin Liu <yuzhixu@nyu.edu, StarLiu@seas.upenn.edu>
"""Framework for SakePP"""


__version__ = "1.0.0"
from .models.sakepp import SAKEPP
from .models.dgn import DGN,DGNLinearFusion
from .dataset.datasets import CustomDGLDataset
from .utils.logging import setup_logging
from .utils.kfold import train_kfold


__all__ = [
    'CustomDGLDataset',
    'train_kfold', 
    'setup_logging',
    'SAKEPP',
    'DGN',
    'DGNLinearFusion',
    '__version__'
    ]


def check_version(min_version):
    from packaging import version
    if version.parse(__version__) < version.parse(min_version):
        raise ImportError(f"sakepp {min_version}+ required")