import numpy as np
import h5py
import time 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List

class MAEPMTDataset(Dataset):
    def __init__(
        self, 
        data_path: str,
        pos_encoding_path: str,
        mask_ratio: float = 0.50,  
        max_waveforms: Optional[int] = 1000,
        dataset_name: str = "unnamed" 
    ):
        self.data_path = data_path
        self.mask_ratio = mask_ratio
        self.dataset_name = dataset_name
        
        # Load position encodings
        pos_data = np.load(pos_encoding_path)
        self.x_encodings = torch.tensor(pos_data['x'], dtype=torch.float32)
        self.y_encodings = torch.tensor(pos_data['y'], dtype=torch.float32)
        self.z_encodings = torch.tensor(pos_data['z'], dtype=torch.float32)
        
        # Load event data
        with h5py.File(data_path, 'r') as f:
            total_waveforms = len(f['charges'])

            # Apply max_waveforms limit if specified
            effective_max = total_waveforms if max_waveforms is None else min(max_waveforms, total_waveforms)
            self.charges = f['charges'][:effective_max]
            self.deltas = f['deltas'][:effective_max]
            self.masks = f['masks'][:effective_max]
            self.pmt_ids = f['pmt_ids'][:effective_max] if 'pmt_ids' in f else None
        
        self.length = len(self.charges)
        print(f"Dataset '{dataset_name}' initialized with {self.length}/{total_waveforms} waveforms (max_limit: {max_waveforms if max_waveforms is not None else 'None'})")
    
    def _create_masks(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Creates prediction masks for valid positions """
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


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        charges = torch.tensor(self.charges[idx], dtype=torch.float32)
        deltas = torch.tensor(self.deltas[idx], dtype=torch.float32)
        attention_mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        
        assert torch.all(charges[~attention_mask] == 0), \
        f"Mask should cover zeros, but found {charges[~attention_mask]} at idx {idx}"
        
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
        
        # Which positions will be masked during training
        pred_mask = self._create_masks(attention_mask)
        
        original_values = torch.zeros((n_pmts, 2), dtype=torch.float32)
        original_values[:, 0] = charges  # q
        original_values[:, 1] = deltas   # dt

        valid_pmts = attention_mask.sum().item()
        masked_pmts = (pred_mask & attention_mask).sum().item()
        unmasked_pmts = (attention_mask & ~pred_mask).sum().item()
        
        #Debugging
        # assert masked_pmts + unmasked_pmts == valid_pmts, \
        # f"Mask mismatch: {masked_pmts} masked + {unmasked_pmts} unmasked != {valid_pmts} valid PMTs"
    
        # if idx % 100 == 0:  # Print only occasionally to avoid flooding
        #     print(f"Sample {idx}: {masked_pmts} masked + {unmasked_pmts} unmasked = {valid_pmts} valid PMTs")

        return {
            "event_tensor": event_tensor,
            "attention_mask": attention_mask,
            "pred_mask": pred_mask,
            "original_values": original_values
        }

    def __len__(self) -> int:
        return self.length

def create_mae_dataloader(
    data_path: str,
    pos_encoding_path: str,
    batch_size: int = 64,
    num_workers: int = 8,
    shuffle: bool = True,
    max_waveforms: Optional[int] = 1000,
    mask_ratio: float = 0.50,
    dataset_name: str = "unnamed"
) -> DataLoader:
    """
    Create a DataLoader using preprocessed data for MAE-style pretraining
    """
    dataset = MAEPMTDataset(
        data_path, 
        pos_encoding_path,
        mask_ratio=mask_ratio,
        max_waveforms=max_waveforms,
        dataset_name=dataset_name
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def split_and_combine_dataloaders(
    cosmic_data_path: str,
    marley_data_path: str,
    pos_encoding_path: str,
    batch_size: int = 64,
    num_workers: int = 8,
    train_split: float = 0.8,
    seed: int = 42,
    mask_ratio: float = 0.50,
    max_waveforms: Optional[int] = 1000  
) -> Tuple[DataLoader, DataLoader]:
    
    # If we want to reproduce the split set same random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Print debug information about max_waveforms
    print(f"Creating dataloaders with max_waveforms={max_waveforms}")
    
    # Load at most max_waveforms per dataset (for both cosmic and marley)
    cosmic_dataset = MAEPMTDataset(
        cosmic_data_path, 
        pos_encoding_path,
        mask_ratio=mask_ratio,
        max_waveforms=max_waveforms,  
        dataset_name="cosmic"
    )
    
    marley_dataset = MAEPMTDataset(
        marley_data_path, 
        pos_encoding_path,
        mask_ratio=mask_ratio,
        max_waveforms=max_waveforms,  
        dataset_name="marley"
    )
    
    # Split each dataset first into train / val then concatenate into full dataset 
    cosmic_train_size = int(len(cosmic_dataset) * train_split)
    cosmic_val_size = len(cosmic_dataset) - cosmic_train_size
    
    marley_train_size = int(len(marley_dataset) * train_split)
    marley_val_size = len(marley_dataset) - marley_train_size
    
    cosmic_train_dataset, cosmic_val_dataset = torch.utils.data.random_split(
        cosmic_dataset, 
        [cosmic_train_size, cosmic_val_size]
    )
    
    marley_train_dataset, marley_val_dataset = torch.utils.data.random_split(
        marley_dataset, 
        [marley_train_size, marley_val_size]
    )
    
    # Combine the train datasets and val datasets
    train_dataset = torch.utils.data.ConcatDataset([cosmic_train_dataset, marley_train_dataset])
    val_dataset = torch.utils.data.ConcatDataset([cosmic_val_dataset, marley_val_dataset])
    
    print(f"Combined training dataset: {len(train_dataset)} events")
    print(f"Combined validation dataset: {len(val_dataset)} events")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader