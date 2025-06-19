from typing import Optional, Tuple, Dict, List
import os
import pandas as pd
import random
import time
import h5py
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch
import torch
from torch.utils.data import Dataset
import warnings
from Bio import BiopythonWarning
from dataclasses import dataclass
import argparse

warnings.simplefilter('ignore', BiopythonWarning)  # Ignore BioPython warnings

@dataclass
class ProcessingConfig:
    """Configuration class for processing parameters"""
    # Distance and threshold parameters
    ca_distance_cutoff: float = 10.0
    min_residue_count: int = 8
    noise_factor: float = 1e-5
    default_dihedral_value: float = 1e-5
    missing_dihedral_value: float = 0.0
    
    # Chain-related parameters
    required_chain_count: int = 2
    default_chain_labels: List[str] = None
    
    # Dataset parameters
    max_samples: int = 84000
    
    # File paths (can be set via parameters)
    paths_file: str = None
    labels_file: str = None
    output_hdf5_path: str = None
    
    def __post_init__(self):
        if self.default_chain_labels is None:
            self.default_chain_labels = ['A', 'B']

# Standard amino acid residue type encoding
def get_residue_type_encoding() -> Dict[str, List[int]]:
    """Get one-hot encoding for standard 20 amino acids"""
    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL']
    
    encoding = {}
    for i, aa in enumerate(amino_acids):
        vector = [0] * len(amino_acids)
        vector[i] = 1
        encoding[aa] = vector
    
    return encoding

# Element to atomic number mapping
def get_element_to_atomic_number() -> Dict[str, int]:
    """Get mapping of common elements to atomic numbers"""
    return {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 
        'CL': 17, 'BR': 35, 'I': 53
    }

def _angle(r0: torch.Tensor, r1: torch.Tensor) -> torch.Tensor:
    """Calculate angle between two vectors"""
    angle = torch.atan2(
        torch.norm(torch.cross(r0, r1), p=2, dim=-1),
        torch.sum(torch.mul(r0, r1), dim=-1),
    )
    return angle

def dihedral(
    x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
    noise_factor: float = 1e-5
) -> torch.Tensor:
    """
    Calculate dihedral angle between four points
    
    Reference:
    Closely follows implementation in Yutong Zhao's timemachine:
    https://github.com/proteneer/timemachine/blob/1a0ab45e605dc1e28c44ea90f38cb0dedce5c4db/timemachine/potentials/bonded.py#L152-L199
    """
    assert x0.shape == x1.shape == x2.shape == x3.shape

    # Calculate displacement vectors 0->1, 2->1, 2->3
    r01 = x1 - x0 + torch.randn_like(x0) * noise_factor
    r21 = x1 - x2 + torch.randn_like(x0) * noise_factor
    r23 = x3 - x2 + torch.randn_like(x0) * noise_factor

    # Calculate normal vectors
    n1 = torch.cross(r01, r21)
    n2 = torch.cross(r21, r23)

    rkj_normed = r21 / torch.norm(r21, dim=-1, keepdim=True)

    y = torch.sum(torch.mul(torch.cross(n1, n2), rkj_normed), dim=-1)
    x = torch.sum(torch.mul(n1, n2), dim=-1)

    # Choose correct quadrant
    theta = torch.atan2(y, x)
    return theta

def calculate_dihedral_angles(
    residue, 
    config: ProcessingConfig
) -> Tuple[float, float]:
    """
    Calculate phi and psi dihedral angles for a residue
    
    Args:
        residue: BioPython residue object
        config: Processing configuration
        
    Returns:
        phi, psi dihedral angles (in degrees)
    """
    try:
        N_pos = residue['N'].get_coord()
        CA_pos = residue['CA'].get_coord()
        C_pos = residue['C'].get_coord()

        residue_id = residue.get_id()
        chain = residue.get_parent()
        
        # Get previous and next residues
        prev_residue = None
        next_residue = None
        
        try:
            prev_residue = chain[residue_id[1] - 1] if residue_id[1] > 0 else None
        except KeyError:
            pass
            
        try:
            next_residue = chain[residue_id[1] + 1] if residue_id[1] < len(chain) - 1 else None
        except KeyError:
            pass

        # Calculate phi angle
        if prev_residue is not None:
            try:
                prev_C_pos = prev_residue['C'].get_coord()
                phi = dihedral(
                    torch.tensor(prev_C_pos), 
                    torch.tensor(N_pos), 
                    torch.tensor(CA_pos), 
                    torch.tensor(C_pos),
                    config.noise_factor
                ).item() * 180 / np.pi
            except KeyError:
                phi = config.default_dihedral_value
        else:
            phi = config.default_dihedral_value

        # Calculate psi angle
        if next_residue is not None:
            try:
                next_N_pos = next_residue['N'].get_coord()
                psi = dihedral(
                    torch.tensor(N_pos), 
                    torch.tensor(CA_pos), 
                    torch.tensor(C_pos), 
                    torch.tensor(next_N_pos),
                    config.noise_factor
                ).item() * 180 / np.pi
            except KeyError:
                psi = config.default_dihedral_value
        else:
            psi = config.default_dihedral_value
            
        return phi, psi

    except KeyError:
        # If key atoms are missing, return default values
        return config.missing_dihedral_value, config.missing_dihedral_value

def read_pdb_interface_atoms_ca_distance(
    path: str, 
    config: ProcessingConfig,
    residue_encoding: Dict[str, List[int]] = None,
    element_mapping: Dict[str, int] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Read PDB file and extract interface atom information
    
    Args:
        path: PDB file path
        config: Processing configuration
        residue_encoding: Residue type encoding dictionary
        element_mapping: Element to atomic number mapping
        
    Returns:
        Tuple containing coordinates, atom IDs, residue types, chain IDs, CA flags, phi angles, psi angles
        Returns None if processing fails
    """
    if residue_encoding is None:
        residue_encoding = get_residue_type_encoding()
    if element_mapping is None:
        element_mapping = get_element_to_atomic_number()
        
    parser = PDBParser()
    try:
        structure = parser.get_structure('PDB_structure', path)
    except Exception as e:
        print(f"Skipping {path}: Cannot parse PDB file - {e}")
        return None
    
    coordinates, atom_ids, residue_types, chain_ids, ca_flags, phi_angles, psi_angles = [], [], [], [], [], [], []

    selected_chains = [chain for chain in structure.get_chains()][:config.required_chain_count]
    if len(selected_chains) != config.required_chain_count:
        print(f"Skipping {path}: PDB file must contain exactly {config.required_chain_count} chains, found {len(selected_chains)}.")
        return None

    # Create chain ID mapping
    chain_id_map = {}
    for i, chain in enumerate(selected_chains):
        if i < len(config.default_chain_labels):
            chain_id_map[chain.id] = config.default_chain_labels[i]
        else:
            chain_id_map[chain.id] = chr(ord('A') + i)

    # Get all CA atoms
    ca_atoms_by_chain = {}
    all_ca_atoms = []
    for chain in selected_chains:
        ca_atoms = [atom for atom in chain.get_atoms() if atom.id == 'CA']
        ca_atoms_by_chain[chain.id] = ca_atoms
        all_ca_atoms.extend(ca_atoms)
    
    if len(all_ca_atoms) == 0:
        print(f"Skipping {path}: No CA atoms found")
        return None
        
    ns = NeighborSearch(all_ca_atoms)

    # Find CA atom pairs within cutoff distance
    ca_pairs = []
    for i, chain_a in enumerate(selected_chains):
        for j, chain_b in enumerate(selected_chains):
            if i >= j:  # Avoid duplicates and self-comparison
                continue
                
            for ca_a in ca_atoms_by_chain[chain_a.id]:
                neighbors = ns.search(ca_a.coord, config.ca_distance_cutoff, level='R')
                for residue_b in neighbors:
                    if residue_b.get_full_id()[2] == chain_b.id:
                        for atom in residue_b:
                            if atom.id == 'CA':
                                ca_pairs.append((ca_a, atom))
                                break

    if len(ca_pairs) == 0:
        print(f"Skipping {path}: No CA atom pairs found within {config.ca_distance_cutoff}Å")
        return None

    missing_atom_count = 0
    valid_residue_count = 0

    for ca_a, ca_b in set(ca_pairs):
        residue_a = ca_a.get_parent()
        residue_b = ca_b.get_parent()

        # Process residues from both chains
        for residue in [residue_a, residue_b]:
            chain_id = chain_id_map[residue.get_parent().id]
            valid_residue_count += 1
            
            # Process all atoms in the residue
            for atom in residue:
                coordinates.append(atom.coord)
                atom_ids.append(element_mapping.get(atom.element, 0))
                
                res_type = residue.get_resname()
                if res_type not in residue_encoding:
                    print(f"Skipping {path}: Unknown residue type {res_type}")
                    return None
                residue_types.append(res_type)
                chain_ids.append(ord(chain_id))
                ca_flags.append(1 if atom.id == 'CA' else 0)

            # Calculate phi and psi dihedral angles
            phi, psi = calculate_dihedral_angles(residue, config)
            if phi == config.missing_dihedral_value:
                missing_atom_count += 1
                
            phi_angles.extend([phi] * len(residue))
            psi_angles.extend([psi] * len(residue))

    if missing_atom_count > 0:
        print(f"Warning: {missing_atom_count} residues in {path} are missing atoms for dihedral calculation")

    if valid_residue_count < config.min_residue_count:
        print(f"Skipping {path}: Only {valid_residue_count} residues (less than {config.min_residue_count})")
        return None

    return (
        torch.tensor(coordinates, dtype=torch.float),
        torch.tensor(atom_ids, dtype=torch.long),
        torch.tensor([residue_encoding[res_type] for res_type in residue_types], dtype=torch.float),
        torch.tensor(chain_ids, dtype=torch.long),
        torch.tensor(ca_flags, dtype=torch.long),
        torch.tensor(phi_angles, dtype=torch.float),
        torch.tensor(psi_angles, dtype=torch.float)
    )

class InterfaceDataset(Dataset):
    """Protein interface dataset class"""
    
    def __init__(self, paths: List[str], labels: List[float], config: ProcessingConfig):
        self.paths = paths
        self.labels = labels
        self.config = config
        self.residue_encoding = get_residue_type_encoding()
        self.element_mapping = get_element_to_atomic_number()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        pdb_data = read_pdb_interface_atoms_ca_distance(
            self.paths[idx], 
            self.config,
            self.residue_encoding,
            self.element_mapping
        )
        if pdb_data is None:
            return None
        return (*pdb_data, self.labels[idx])

def read_file_to_list(filepath: str) -> List[str]:
    """Read file and return content as list"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with open(filepath, 'r') as file:
        return [line.strip() for line in file.readlines() if line.strip()]

def save_to_hdf5_with_index(
    dataset: InterfaceDataset, 
    hdf5_path: str, 
    max_samples: int = None
) -> None:
    """
    Save dataset to HDF5 file
    
    Args:
        dataset: Dataset object
        hdf5_path: Output HDF5 file path
        max_samples: Maximum number of samples, None for unlimited
    """
    if max_samples is None:
        max_samples = len(dataset)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    
    with h5py.File(hdf5_path, 'w') as f:
        # Create datasets
        datasets = {
            'coordinates': f.create_dataset("coordinates", shape=(0, 3), maxshape=(None, 3), 
                                          dtype='float32', chunks=True, compression="gzip"),
            'atom_ids': f.create_dataset("atom_ids", shape=(0,), maxshape=(None,), 
                                       dtype='int64', chunks=True, compression="gzip"),
            'residue_types': f.create_dataset("residue_types", shape=(0, 20), maxshape=(None, 20), 
                                            dtype='float32', chunks=True, compression="gzip"),
            'chain_ids': f.create_dataset("chain_ids", shape=(0,), maxshape=(None,), 
                                        dtype='int64', chunks=True, compression="gzip"),
            'ca_flags': f.create_dataset("ca_flags", shape=(0,), maxshape=(None,), 
                                       dtype='int64', chunks=True, compression="gzip"),
            'phi_angles': f.create_dataset("phi_angles", shape=(0,), maxshape=(None,), 
                                         dtype='float32', chunks=True, compression="gzip"),
            'psi_angles': f.create_dataset("psi_angles", shape=(0,), maxshape=(None,), 
                                         dtype='float32', chunks=True, compression="gzip"),
            'labels': f.create_dataset("labels", shape=(0,), maxshape=(None,), 
                                     dtype='float32', chunks=True, compression="gzip"),
            'indices': f.create_dataset("indices", shape=(0, 2), maxshape=(None, 2), 
                                      dtype='int64', chunks=True, compression="gzip")
        }

        coord_index = 0
        valid_sample_count = 0
        
        for idx, data in enumerate(dataset):
            if data is None:
                print(f"Skipping sample {idx} (invalid data)")
                continue

            coordinates, atom_ids, residue_types, chain_ids, ca_flags, phi_angles, psi_angles, label = data
            n = coordinates.size(0)
            
            valid_sample_count += 1
            new_size = coord_index + n
            
            # Resize datasets
            for key in ['coordinates', 'atom_ids', 'residue_types', 'chain_ids', 
                       'ca_flags', 'phi_angles', 'psi_angles']:
                if key == 'coordinates':
                    datasets[key].resize(new_size, axis=0)
                elif key == 'residue_types':
                    datasets[key].resize(new_size, axis=0)
                else:
                    datasets[key].resize(new_size, axis=0)
            
            datasets['labels'].resize(valid_sample_count, axis=0)
            datasets['indices'].resize(valid_sample_count, axis=0)

            # Write data
            datasets['coordinates'][coord_index:coord_index + n] = coordinates.numpy()
            datasets['atom_ids'][coord_index:coord_index + n] = atom_ids.numpy()
            datasets['residue_types'][coord_index:coord_index + n] = residue_types.numpy()
            datasets['chain_ids'][coord_index:coord_index + n] = chain_ids.numpy()
            datasets['ca_flags'][coord_index:coord_index + n] = ca_flags.numpy()
            datasets['phi_angles'][coord_index:coord_index + n] = phi_angles.numpy()
            datasets['psi_angles'][coord_index:coord_index + n] = psi_angles.numpy()
            datasets['labels'][valid_sample_count - 1] = label
            datasets['indices'][valid_sample_count - 1] = [coord_index, coord_index + n]
            
            coord_index += n

            if valid_sample_count >= max_samples:
                print(f"Reached maximum number of samples ({max_samples}), stopping processing.")
                break

def retrieve_sample(hdf5_path: str, sample_idx: int) -> Tuple[torch.Tensor, ...]:
    """Retrieve specified sample from HDF5 file"""
    with h5py.File(hdf5_path, 'r') as f:
        start, end = f['indices'][sample_idx]
        coordinates = torch.tensor(f['coordinates'][start:end])
        atom_ids = torch.tensor(f['atom_ids'][start:end])
        residue_types = torch.tensor(f['residue_types'][start:end])
        chain_ids = torch.tensor(f['chain_ids'][start:end])
        ca_flags = torch.tensor(f['ca_flags'][start:end])
        phi_angles = torch.tensor(f['phi_angles'][start:end])
        psi_angles = torch.tensor(f['psi_angles'][start:end])
        label = torch.tensor(f['labels'][sample_idx])
    return coordinates, atom_ids, residue_types, chain_ids, ca_flags, phi_angles, psi_angles, label

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description='Process PDB files and generate HDF5 dataset',
        epilog='''
Examples:
  # Basic usage
  python hdf.py --paths-file pdb_paths.txt --labels-file labels.txt --output dataset.h5
  
  # With custom parameters
  python hdf.py --paths-file pdb_paths.txt --labels-file labels.txt --output dataset.h5 --max-samples 1000 --cutoff 8.0 --min-residues 5
  
  # Test with single sample
  python hdf.py --paths-file pdb_paths.txt --labels-file labels.txt --output test.h5 --max-samples 1
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--paths-file', 
                       required=True, 
                       help='Path to file containing PDB file paths (one path per line)')
    parser.add_argument('--labels-file', 
                       required=True, 
                       help='Path to file containing labels (one label per line)')
    parser.add_argument('--output', 
                       required=True, 
                       help='Output HDF5 file path')
    
    # Optional arguments with defaults
    parser.add_argument('--max-samples', 
                       type=int, 
                       default=84000, 
                       help='Maximum number of samples to process (default: 84000)')
    parser.add_argument('--cutoff', 
                       type=float, 
                       default=10.0, 
                       help='CA distance cutoff in Angstroms (default: 10.0)')
    parser.add_argument('--min-residues', 
                       type=int, 
                       default=8, 
                       help='Minimum number of residues required (default: 8)')
    parser.add_argument('--noise-factor', 
                       type=float, 
                       default=1e-5, 
                       help='Noise factor for dihedral calculation (default: 1e-5)')
    parser.add_argument('--verbose', 
                       action='store_true', 
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.paths_file):
        print(f"Error: Paths file not found: {args.paths_file}")
        return 1
    
    if not os.path.exists(args.labels_file):
        print(f"Error: Labels file not found: {args.labels_file}")
        return 1
    
    # Create configuration
    config = ProcessingConfig(
        ca_distance_cutoff=args.cutoff,
        min_residue_count=args.min_residues,
        max_samples=args.max_samples,
        noise_factor=args.noise_factor
    )
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  CA distance cutoff: {config.ca_distance_cutoff}Å")
        print(f"  Minimum residues: {config.min_residue_count}")
        print(f"  Maximum samples: {config.max_samples}")
        print(f"  Noise factor: {config.noise_factor}")
    
    try:
        # Read data
        if args.verbose:
            print(f"Reading paths from: {args.paths_file}")
        pdb_paths = read_file_to_list(args.paths_file)
        
        if args.verbose:
            print(f"Reading labels from: {args.labels_file}")
        pdb_labels = [float(label) for label in read_file_to_list(args.labels_file)]
        
        if len(pdb_paths) != len(pdb_labels):
            raise ValueError(f"Number of paths ({len(pdb_paths)}) does not match number of labels ({len(pdb_labels)})")
        
        if args.verbose:
            print(f"Loaded {len(pdb_paths)} PDB paths and {len(pdb_labels)} labels")
        
        # Create dataset and save
        dataset = InterfaceDataset(pdb_paths, pdb_labels, config)
        
        print(f"Processing up to {min(args.max_samples, len(dataset))} samples...")
        save_to_hdf5_with_index(dataset, args.output, args.max_samples)
        
        # Test reading first sample if file exists and has data
        if os.path.exists(args.output):
            try:
                with h5py.File(args.output, 'r') as f:
                    if len(f['labels']) > 0:
                        sample_data = retrieve_sample(args.output, 0)
                        print(f"✓ Successfully saved dataset to {args.output}")
                        print(f"✓ First sample contains {sample_data[0].shape[0]} atoms")
                        print(f"✓ Total samples in dataset: {len(f['labels'])}")
                    else:
                        print(f"⚠ Dataset saved but contains no valid samples")
            except Exception as e:
                print(f"⚠ Dataset saved but error reading sample: {e}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
