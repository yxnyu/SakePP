import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def angle_to_3d(phi, psi):
    """Convert dihedral angles to 3D coordinates"""
    phi = np.deg2rad(phi)
    psi = np.deg2rad(psi)
    x = np.sin(phi) * np.cos(psi)
    y = np.sin(phi) * np.sin(psi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def create_edge_index_and_features(ca_coordinates, ca_residue_types, ca_chain_ids, cutoff=10.0, k=30, device='cpu'):
    """Create edge index and features for cross-chain interactions with RBF encoding"""
    num_residues = ca_coordinates.size(0)
    distances = torch.cdist(ca_coordinates, ca_coordinates)

    if torch.isnan(distances).any() or torch.isinf(distances).any():
        logger.error("NaN or Inf values found in distance matrix calculation.")
        return None, None
    
    src_list = []
    dst_list = []
    edge_features = []
    
    for i in range(num_residues):
        current_chain_id = ca_chain_ids[i].item()
        mask = (ca_chain_ids != current_chain_id) & (distances[i] <= cutoff)
        valid_indices = torch.nonzero(mask, as_tuple=True)[0]

        if valid_indices.size(0) > 0:
            sorted_indices = valid_indices[torch.argsort(distances[i, valid_indices])][:k]
            for j in sorted_indices:
                src_list.append(i)
                dst_list.append(j)

                d = distances[i, j]
                edge_attr_temp = [d]

                # Use RBF function to calculate RBF encoding
                D_count = 16
                D_min = 0.
                D_max = cutoff
                D_mu = torch.linspace(D_min, D_max, D_count, device=device)
                D_mu = D_mu.view([1, -1])
                D_sigma = (D_max - D_min) / D_count
                D_expand = d.unsqueeze(-1)
                RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
                edge_attr_temp.extend(RBF.squeeze().tolist())

                edge_features.append(edge_attr_temp)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
        logger.error("NaN or Inf values found when creating edge indices.")
        return None, None

    if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
        logger.error("NaN or Inf values found when creating edge features.")
        return None, None

    return edge_index, edge_attr
