import numpy as np
import torch


def angle_to_3d(phi, psi):
    """Convert dihedral angles to 3D coordinates"""
    phi = np.deg2rad(phi)
    psi = np.deg2rad(psi)
    x = np.sin(phi) * np.cos(psi)
    y = np.sin(phi) * np.sin(psi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def create_edge_index_and_features(coordinates, residue_types, chain_ids, cutoff=10.0, k=30):
    """Create edge index and features"""
    # Calculate distance matrix
    distances = torch.cdist(coordinates, coordinates)

    # Create adjacency matrix
    adj_matrix = (distances < cutoff)

    # Find k nearest neighbors for each node
    k = min(k, adj_matrix.size(0) - 1)
    _, topk_indices = torch.topk(distances, k=k+1, dim=1, largest=False)
    topk_indices = topk_indices[:, 1:]  # Remove self-connection

    # Create edge index
    rows = torch.arange(adj_matrix.size(0)).unsqueeze(1).expand(-1, k)
    edge_index = torch.stack([rows.flatten(), topk_indices.flatten()])

    # Calculate edge features
    src_coords = coordinates[edge_index[0]]
    dst_coords = coordinates[edge_index[1]]
    edge_vectors = dst_coords - src_coords
    edge_lengths = torch.norm(edge_vectors, dim=1, keepdim=True)
    edge_directions = edge_vectors / (edge_lengths + 1e-6)

    # Residue type features
    src_res_types = residue_types[edge_index[0]]
    dst_res_types = residue_types[edge_index[1]]

    # Chain ID features
    same_chain = (chain_ids[edge_index[0]] == chain_ids[edge_index[1]]).float().unsqueeze(1)
    # Combine all edge features
    edge_features = torch.cat([
        edge_lengths,          # 1D: distance
        edge_directions,       # 3D: direction vector
        src_res_types,        # 1D: source node residue type
        dst_res_types,        # 1D: destination node residue type
        same_chain,           # 1D: whether same chain
    ], dim=1)

    return edge_index, edge_features
