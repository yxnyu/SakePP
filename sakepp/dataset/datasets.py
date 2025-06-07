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
            save_path='/scratch/yx2892/GNN/sing_egnn/process/process_decoy_100.pt'
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
                start, end = f['indices'][i]
                if end - start < 1:
                    print(f"Skipping index {i} because it has 1 or fewer atoms.")
                    continue

                # Load data
                coordinates = torch.tensor(f['coordinates'][start:end], dtype=torch.float)
                residue_types = torch.tensor(f['residue_types'][start:end], dtype=torch.float)
                chain_ids = torch.tensor(f['chain_ids'][start:end], dtype=torch.long)
                ca_flags = torch.tensor(f['ca_flags'][start:end], dtype=torch.long)
                phi_angles = torch.tensor(f['phi_angles'][start:end], dtype=torch.float)
                psi_angles = torch.tensor(f['psi_angles'][start:end], dtype=torch.float)

                # Data validation
                if torch.isnan(coordinates).any() or torch.isinf(coordinates).any():
                    self.logger.error(f"Index {i} contains NaN or Inf values in coordinates.")
                    continue
    
                if torch.isnan(residue_types).any() or torch.isinf(residue_types).any():
                    self.logger.error(f"Index {i} contains NaN or Inf values in residue types.")
                    continue
    
                if torch.isnan(phi_angles).any() or torch.isinf(phi_angles).any():
                    self.logger.error(f"Index {i} contains NaN or Inf values in phi angles.")
                    continue
    
                if torch.isnan(psi_angles).any() or torch.isinf(psi_angles).any():
                    self.logger.error(f"Index {i} contains NaN or Inf values in psi angles.")
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
                graph = dgl.graph(([], []), num_nodes=num_nodes)

                graph.add_edges(edge_index[0], edge_index[1])
                graph.edata['attr'] = edge_attr
                graph.ndata['feat'] = node_features
                graph.ndata['coord'] = ca_coordinates

                # Validate graph structure
                if graph.number_of_nodes() != node_features.size(0):
                    print(f"Problem found when processing index {i}: node count mismatch with feature count.")
                    print(f"Expected node count: {node_features.size(0)}, actual node count: {graph.number_of_nodes()}")
                    raise ValueError("Node count mismatch with feature count: {} vs {}".format(graph.number_of_nodes(), node_features.size(0)))

                # Add label
                label = torch.tensor(f['labels'][i], dtype=torch.float)

                # Save graph and label
                self.graphs.append((graph, label))

            self.logger.info("Data processing and saving completed.")
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

