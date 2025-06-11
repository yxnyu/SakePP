# HDF Data Preprocessing Tool Usage Guide

This tool processes PDB files and generates HDF5 datasets for protein interface analysis and machine learning training.

## Overview

The script converts multiple PDB files into a unified HDF5 dataset, extracting protein interface atom information including:
- Atomic coordinates
- Residue type encodings
- Dihedral angle information (phi/psi angles)
- Chain identifiers
- CA atom flags
- Corresponding label data

## Basic Usage

```bash
python hdf.py --paths-file <PDB_paths_file> --labels-file <labels_file> --output <output_HDF5_file>
```

## Primary Usage Example

```bash
python hdf.py --paths-file '/scratch/yx2892/Hbond/ZDock/control/2024pdb/pdb_paths_2.txt' \
              --labels-file '/scratch/yx2892/Hbond/ZDock/control/2024pdb/IRMSD_only.txt' \
              --output '/scratch/yx2892/Hbond/ZDock/control/2024pdb/2024pdb_custom.h5' \
              --max-samples 5000 \
              --cutoff 8.0 \
              --min-residues 5 \
              --verbose
```

## Command Line Arguments

### Required Arguments

- `--paths-file`: Path to file containing PDB file paths (one path per line)
- `--labels-file`: Path to file containing labels (one label per line, corresponding to PDB files)
- `--output`: Output HDF5 file path

### Optional Arguments

- `--max-samples`: Maximum number of samples to process (default: 84000)
- `--cutoff`: CA atom distance cutoff in Angstroms (default: 10.0)
- `--min-residues`: Minimum number of residues required (default: 8)
- `--noise-factor`: Noise factor for dihedral calculation (default: 1e-5)
- `--verbose`: Print verbose output information

## Additional Usage Examples

### Basic usage with minimal parameters:
```bash
python hdf.py --paths-file pdb_paths.txt \
              --labels-file labels.txt \
              --output dataset.h5
```

### Test mode (process single sample):
```bash
python hdf.py --paths-file pdb_paths.txt \
              --labels-file labels.txt \
              --output test.h5 \
              --max-samples 1 \
              --verbose
```

### High-throughput processing:
```bash
python hdf.py --paths-file large_dataset_paths.txt \
              --labels-file large_dataset_labels.txt \
              --output large_dataset.h5 \
              --max-samples 50000 \
              --cutoff 12.0 \
              --min-residues 10 \
              --verbose
```

## Input File Formats

### PDB Paths File Format
One PDB file path per line:
```
/path/to/structure1.pdb
/path/to/structure2.pdb
/path/to/structure3.pdb
```

### Labels File Format
One numerical label per line, corresponding to each PDB file:
```
0.123
0.456
0.789
```

**Note**: The paths file and labels file must have the same number of lines and correspond in order.

## Output HDF5 File Structure

The generated HDF5 file contains the following datasets:

- `coordinates`: Atomic coordinates (shape: [N, 3], dtype: float32)
- `atom_ids`: Atomic type identifiers (shape: [N], dtype: int64)
- `residue_types`: Residue type one-hot encoding (shape: [N, 20], dtype: int64)
- `chain_ids`: Chain identifiers (shape: [N], dtype: int64)
- `ca_flags`: CA atom flags (shape: [N], dtype: int64)
- `phi_angles`: Phi dihedral angles (shape: [N], dtype: float32)
- `psi_angles`: Psi dihedral angles (shape: [N], dtype: float32)
- `labels`: Sample labels (shape: [M], dtype: float32)
- `indices`: Sample index ranges (shape: [M, 2], dtype: int64)

Where N is the total number of atoms across all samples, and M is the number of valid samples.

## Processing Parameters

### Distance Cutoff (--cutoff)
- Controls the distance threshold between CA atoms
- Default value: 10.0 Å
- Only residues with CA atoms closer than this value are included in the interface

### Minimum Residues (--min-residues)
- Minimum number of residues required per sample
- Default value: 8
- Samples with fewer residues than this value will be skipped

### Maximum Samples (--max-samples)
- Maximum number of samples to process
- Default value: 84000
- Can be set to smaller values for testing

## Residue Type Encoding

The tool supports one-hot encoding for 20 standard amino acids:
ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL

## Error Handling and Troubleshooting

### Common Errors and Solutions

1. **File Path Errors**
   ```
   Error: Paths file not found: /path/to/file
   ```
   - Check if the paths file exists and is readable

2. **Label Count Mismatch**
   ```
   Error: Number of paths (100) does not match number of labels (99)
   ```
   - Ensure the paths file and labels file have the same number of lines

3. **Invalid PDB Files**
   ```
   Skipping sample X (invalid data)
   ```
   - Check if PDB files are corrupted or improperly formatted
   - Ensure PDB files contain required atomic information

4. **Insufficient Memory**
   - Reduce the `--max-samples` parameter value
   - Process large datasets in batches

### Performance Optimization Tips

1. **Use SSD Storage**: Improve I/O performance
2. **Adequate Memory**: Large datasets require sufficient RAM
3. **Batch Processing**: Consider batch processing for very large datasets
4. **Parallel Processing**: Run multiple instances to process different data subsets

## Data Validation

After processing completes, the script automatically validates output:
- Checks if HDF5 file was successfully created
- Reads the first sample for validation
- Reports total number of processed samples and atoms

## Dependencies

Ensure the following dependencies are installed:
```bash
pip install numpy pandas torch h5py biopython
```

## Memory Usage

Memory usage depends on:
- PDB file sizes
- Number of samples processed
- Interface complexity

For large datasets, it is recommended to:
- Monitor memory usage
- Process in batches when necessary
- Use 64-bit Python environment

## Performance Monitoring

Use the `--verbose` flag to get detailed processing information:
- Configuration parameter display
- Processing progress
- Error details
- Final statistics
