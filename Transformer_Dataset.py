import torch
import numpy as np
import tables
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import os
from torch import nn, optim
from tqdm import tqdm
import h5py

class PreprocessedPMTDataset(Dataset):
    def __init__(
        self, 
        data_path: str,
        pos_encoding_path: str,
        mask_ratio: float = 0.30,
        max_waveforms: Optional[int] = None
    ):
        self.data_path = data_path
        self.mask_ratio = mask_ratio
        
        # Load position encodings
        pos_data = np.load(pos_encoding_path)
        self.x_encodings = torch.tensor(pos_data['x'], dtype=torch.float32)
        self.y_encodings = torch.tensor(pos_data['y'], dtype=torch.float32)
        self.z_encodings = torch.tensor(pos_data['z'], dtype=torch.float32)
        
        # Load event data
        with h5py.File(data_path, 'r') as f:
            self.charges = f['charges'][:]
            self.deltas = f['deltas'][:]
            self.masks = f['masks'][:]
            
            if max_waveforms is not None:
                self.charges = self.charges[:max_waveforms]
                self.deltas = self.deltas[:max_waveforms]
                self.masks = self.masks[:max_waveforms]
        
        self.length = len(self.charges)
        print(f"Dataset initialized with {self.length} waveforms")
    
    def _create_masks(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Creates prediction masks for valid positions"""
        valid_positions = attention_mask.nonzero()
        if valid_positions.numel() == 0:
            return torch.zeros_like(attention_mask)
            
        if len(valid_positions.shape) > 1:
            valid_positions = valid_positions.squeeze()
            if len(valid_positions.shape) == 0:
                valid_positions = valid_positions.unsqueeze(0)
        
        n_valid = valid_positions.size(0)
        n_to_mask = max(1, int(n_valid * self.mask_ratio))
        
        if n_valid == 1:
            pred_mask = torch.zeros_like(attention_mask)
            pred_mask[valid_positions] = True
            return pred_mask
        
        perm = torch.randperm(n_valid)
        mask_indices = perm[:n_to_mask]
        selected_positions = valid_positions[mask_indices]
        
        pred_mask = torch.zeros_like(attention_mask)
        pred_mask[selected_positions] = True
        
        return pred_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get data for this event
        charges = torch.tensor(self.charges[idx], dtype=torch.float32)
        deltas = torch.tensor(self.deltas[idx], dtype=torch.float32)
        attention_mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        
        # Create input tensor
        x_y_dim = self.x_encodings.shape[1]  # Should be 8
        total_dim = 2 * x_y_dim + 3  # 8 for x, 8 for y, 1 for z, 1 for q, 1 for dt
        
        n_pmts = len(charges)
        event_tensor = torch.zeros((n_pmts, total_dim), dtype=torch.float32)
        
        # Fill in the tensor
        event_tensor[:, 0] = charges
        event_tensor[:, 1:x_y_dim+1] = self.x_encodings
        event_tensor[:, x_y_dim+1:2*x_y_dim+1] = self.y_encodings
        event_tensor[:, 2*x_y_dim+1:2*x_y_dim+2] = self.z_encodings
        event_tensor[:, 2*x_y_dim+2] = deltas
        
        # Create prediction mask
        pred_mask = self._create_masks(attention_mask)
        
        return event_tensor, attention_mask, pred_mask

    def __len__(self) -> int:
        return self.length

def create_dataloader(
    data_path: str,
    pos_encoding_path: str,
    batch_size: int = 64,
    num_workers: int = 8,
    shuffle: bool = True,
    max_waveforms: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader using preprocessed data
    """
    dataset = PreprocessedPMTDataset(
        data_path, 
        pos_encoding_path,
        max_waveforms=max_waveforms
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )