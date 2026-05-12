import argparse
import gym
import random
import numpy as np
import copy
# import d4rl
from scipy.spatial import cKDTree
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
import copy

import h5py
from datetime import datetime
import os
import shutil

def setup_output_directory(merge_name, base_dir="dataset_analysis"):
    """Create timestamped output directory and return its path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def find_h5_files(directory):
    """Find all .h5 files in the given directory"""
    h5_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files

def merge_h5_files(h5_files, output_dir):
    """Merge multiple HDF5 files into one temporary file"""
    merged_file_path = os.path.join(output_dir, "merged_temp.h5")
    
    with h5py.File(merged_file_path, 'w') as merged_file:
        # Initialize datasets based on first file
        with h5py.File(h5_files[0], 'r') as first_file:
            for key in first_file.keys():
                shape = list(first_file[key].shape)
                shape[0] = 0  # Will grow as we append
                maxshape = tuple([None] + shape[1:]) if len(shape) > 1 else (None,)
                dtype = first_file[key].dtype
                merged_file.create_dataset(key, shape=tuple(shape), maxshape=maxshape, 
                                        dtype=dtype, chunks=True)
        
        # Append data from all files
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as src_file:
                for key in src_file.keys():
                    # Resize merged dataset
                    merged_dataset = merged_file[key]
                    current_size = merged_dataset.shape[0]
                    new_size = current_size + src_file[key].shape[0]
                    merged_dataset.resize(new_size, axis=0)
                    
                    # Append data
                    merged_dataset[current_size:new_size] = src_file[key][:]
    
    return merged_file_path

def get_trajs(input_dir):
    output_dir = setup_output_directory(input_dir.replace('/', '_'))
    try:
        h5_files = find_h5_files(input_dir)
        npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        if not h5_files and not npy_files:
            raise ValueError(f"No .h5 or .npy files found in directory: {input_dir}")
        if h5_files and npy_files:
            assert 0, "Error: Both .h5 and .npy files found. Defaulting to .h5 analysis."
        
        obs, acs, rew = None, None, None
        
        if h5_files:
            # Process HDF5 files
            print(f"Found {len(h5_files)} .h5 files:")
            for file in h5_files:
                print(f"  {file}")
            merged_file_path = merge_h5_files(h5_files, output_dir)
            with h5py.File(merged_file_path, 'r') as source_file:
                obs = source_file['obs'][:]
                acs = source_file['actions'][:]
                rew = source_file['reward'][:]
                dones = source_file['terminated'][:]
                
                if 'filled' not in source_file:
                    traj_lengths = [source_file['obs'].shape[1]] * source_file['obs'].shape[0]
                else:
                    filled = source_file['filled'][:]  # [bs, seq_len]
                    traj_lengths = np.sum(filled, axis=1).tolist()
                
                if rew is not None:
                    if 'filled' in source_file:
                        rew = rew * filled
                    returns = np.sum(rew[:, :, 0], axis=1).tolist()
                else:
                    print("\nWarning: 'reward' key not found in dataset. Skipping return analysis.")
                    returns = None
            
            # Clean up temporary merged file
            shutil.rmtree(output_dir)
            
        else:
            # Process NumPy arrays
            print(f"Found {len(npy_files)} .npy files:")
            for file in npy_files:
                print(f"  {file}")
            
            # Load numpy arrays
            data_dict = {}
            for file in npy_files:
                key = file.replace('.npy', '')
                data_dict[key] = np.load(os.path.join(input_dir, file))
            
            print("\nDataset contains:")
            for key, arr in data_dict.items():
                print(f"  {key}: shape {arr.shape}, dtype {arr.dtype}")
            
            obs = data_dict['obs']
            acs = data_dict.get('acs', None)
            rew = data_dict.get('rew', None)[:, :, 0]
            returns = rew[:, :, 0].sum(axis=-1).tolist()
            
    except Exception as e:
        print(f"\nError occurred during analysis: {str(e)}")
        raise
    
    return obs, acs, rew, dones