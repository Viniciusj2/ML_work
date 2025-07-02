import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
import h5py

class ClassificationDataset(Dataset):
    """
    Dataset for binary classification using existing MAE dataset structure
    """
    def __init__(
        self,
        cosmic_data_path: str,
        marley_data_path: str,
        pos_encoding_path: str,
        max_waveforms_per_class: Optional[int] = 1000
    ):
        self.pos_encoding_path = pos_encoding_path
        
        # Load position encodings (same as MAE dataset)
        pos_data = np.load(pos_encoding_path)
        self.x_encodings = torch.tensor(pos_data['x'], dtype=torch.float32)
        self.y_encodings = torch.tensor(pos_data['y'], dtype=torch.float32)
        self.z_encodings = torch.tensor(pos_data['z'], dtype=torch.float32)
        
        # Calculate feature dimensions
        self.spatial_dim = self.x_encodings.shape[1] * 2 + 1
        self.non_spatial_dim = 2
        self.total_dim = self.spatial_dim + self.non_spatial_dim
        
        # Load cosmic data (label = 0)
        with h5py.File(cosmic_data_path, 'r') as f:
            cosmic_count = len(f['charges']) if max_waveforms_per_class is None else min(max_waveforms_per_class, len(f['charges']))
            cosmic_charges = f['charges'][:cosmic_count]
            cosmic_deltas = f['deltas'][:cosmic_count]
            cosmic_masks = f['masks'][:cosmic_count]
        
        # Load marley data (label = 1)
        with h5py.File(marley_data_path, 'r') as f:
            marley_count = len(f['charges']) if max_waveforms_per_class is None else min(max_waveforms_per_class, len(f['charges']))
            marley_charges = f['charges'][:marley_count]
            marley_deltas = f['deltas'][:marley_count]
            marley_masks = f['masks'][:marley_count]
        
        # Combine data
        self.charges = np.concatenate([cosmic_charges, marley_charges], axis=0)
        self.deltas = np.concatenate([cosmic_deltas, marley_deltas], axis=0)
        self.masks = np.concatenate([cosmic_masks, marley_masks], axis=0)
        
        # Create labels (0 for cosmic, 1 for marley)
        self.labels = np.concatenate([
            np.zeros(cosmic_count, dtype=np.float32),
            np.ones(marley_count, dtype=np.float32)
        ])
        
        print(f"Classification dataset: {cosmic_count} cosmic + {marley_count} marley = {len(self.labels)} total")
        print(f"Class balance: {(self.labels == 0).sum()} cosmic, {(self.labels == 1).sum()} marley")
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get raw values
        charges = torch.tensor(self.charges[idx], dtype=torch.float32)
        deltas = torch.tensor(self.deltas[idx], dtype=torch.float32)
        attention_mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Create input tensor (same format as MAE dataset)
        x_y_dim = self.x_encodings.shape[1]
        n_pmts = len(charges)
        
        event_tensor = torch.zeros((n_pmts, self.total_dim), dtype=torch.float32)
        event_tensor[:, 0] = charges
        event_tensor[:, 1:x_y_dim+1] = self.x_encodings
        event_tensor[:, x_y_dim+1:2*x_y_dim+1] = self.y_encodings
        event_tensor[:, 2*x_y_dim+1:2*x_y_dim+2] = self.z_encodings
        event_tensor[:, 2*x_y_dim+2] = deltas
        
        return {
            "event_tensor": event_tensor,
            "attention_mask": attention_mask,
            "label": label
        }
    
    def __len__(self) -> int:
        return len(self.labels)


def create_classification_dataloaders(
    cosmic_data_path: str,
    marley_data_path: str,
    pos_encoding_path: str,
    batch_size: int = 64,
    train_split: float = 0.8,
    num_workers: int = 8,
    max_waveforms_per_class: Optional[int] = 1000,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for binary classification
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create full dataset
    dataset = ClassificationDataset(
        cosmic_data_path,
        marley_data_path,
        pos_encoding_path,
        max_waveforms_per_class
    )
    
    # Split into train/val
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
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
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

