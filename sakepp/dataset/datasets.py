import os
import h5py
import torch
import dgl
import logging
from dgl.data import DGLDataset
from torch.utils.data import Subset
import numpy as np
from .preprocessing import angle_to_3d, create_edge_index_and_features


class CustomDGLDataset(DGLDataset):
    """Custom DGL dataset class"""
    def __init__(
            self, 
            filename, 
            save_path='/weights/process_pdb_0311.pt'
            ):
        self.filename = filename
        self._save_path = save_path
        self.logger = logging.getLogger(f"{__name__}.CustomDGLDataset")

        super().__init__(name='custom_dataset')
    def process(self):
        """Process the dataset"""
        if os.path.exists(self._save_path):
            self.load_processed_data()
        else:
            self.process_and_save_data()

    def process_and_save_data(self):
        """Process and save data"""
        with h5py.File(self.filename, 'r') as f:
            self.graphs = []
            self.labels = []
            num_indices = len(f['indices'])
            
            for i in range(num_indices):
                # try:
                start, end = f['indices'][i]
                if end - start < 1:
                    self.logger.warning(
                        f"Skipping index {i} because it has 1 or fewer atoms."
                        )
                    continue

                # Load data
                coordinates = torch.tensor(f['coordinates'][start:end], dtype=torch.float)
                residue_types = torch.tensor(f['residue_types'][start:end], dtype=torch.float)
                chain_ids = torch.tensor(f['chain_ids'][start:end], dtype=torch.long)
                ca_flags = torch.tensor(f['ca_flags'][start:end], dtype=torch.long)
                phi_angles = torch.tensor(f['phi_angles'][start:end], dtype=torch.float)
                psi_angles = torch.tensor(f['psi_angles'][start:end], dtype=torch.float)

                # Data validation
                tensors_to_check = {
                    'coordinates': coordinates,
                    'residue_types': residue_types,
                    'phi_angles': phi_angles,
                    'psi_angles': psi_angles
                }
                
                for name, tensor in tensors_to_check.items():
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        self.logger.error(f"Index {i} contains NaN or Inf values in {name}.")
                        continue
                
                # Process CA atoms
                ca_mask = ca_flags == 1
                ca_coordinates = coordinates[ca_mask]
                ca_residue_types = residue_types[ca_mask]
                ca_chain_ids = chain_ids[ca_mask]
                ca_phi_angles = phi_angles[ca_mask]
                ca_psi_angles = psi_angles[ca_mask]

                # Create node features
                angle_features = torch.tensor(angle_to_3d(ca_phi_angles, ca_psi_angles), dtype=torch.float)
                node_features = torch.cat([
                    ca_residue_types,
                    angle_features,
                    ca_coordinates
                ], dim=-1)
                
                # Validate node features
                if torch.isnan(node_features).any() or torch.isinf(node_features).any():
                    self.logger.error(f"Index {i} contains NaN or Inf values in node features.")
                    continue
                # Create edge features
                
                edge_index, edge_attr = create_edge_index_and_features(
                    ca_coordinates,
                    ca_residue_types,
                    ca_chain_ids,
                    cutoff=10.0,
                    k=30
                )

                # Validate edge features
                if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                    self.logger.error(f"Index {i} contains NaN or Inf values in edge features.")
                    continue

                # Create graph
                num_nodes = ca_coordinates.size(0)
                graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

                # Add graph features
                graph.edata['attr'] = edge_attr
                graph.ndata['feat'] = node_features
                graph.ndata['coord'] = ca_coordinates

                # Validate graph structure
                if graph.number_of_nodes() != node_features.size(0):
                    error_msg = f"Node count mismatch: {graph.number_of_nodes()} vs {node_features.size(0)}"
                    self.logger.error(error_msg)
                    continue

                # Add label
                label = torch.tensor(f['labels'][i], dtype=torch.float).view(1)

                # Save graph and label
                self.graphs.append((graph, label))

            self.logger.info(f"Data processing completed. Processed {len(self.graphs)} valid samples.")
            torch.save((self.graphs, self.labels), self._save_path)

    def load_processed_data(self):
        """Load processed data"""
        self.logger.info(f"Loading processed data from {self._save_path}")
        self.graphs, self.labels = torch.load(self._save_path)
        self.logger.info(f"Loading completed. Dataset size: {len(self.graphs)}")

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

