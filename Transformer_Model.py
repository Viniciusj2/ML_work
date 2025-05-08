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

class ModifiedTransformerModel(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int):
        super(ModifiedTransformerModel, self).__init__()
     
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        x_y_dim = 8
        
        self.q_expander = nn.Linear(1, embedding_dim)
        self.dt_expander = nn.Linear(1, embedding_dim)
        self.x_encoder = nn.Linear(x_y_dim, embedding_dim)
        self.y_encoder = nn.Linear(x_y_dim, embedding_dim)
        self.z_encoder = nn.Linear(1, 1)
        
        # mask embedding for when we mask q and dt
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.q_decoder = nn.Linear(embedding_dim, 1)
        self.dt_decoder = nn.Linear(embedding_dim, 1)
    
    def forward(self,x: torch.Tensor,attention_mask: torch.Tensor,pred_mask: torch.Tensor = None,mask_ratio: float = 1.0) -> torch.Tensor:
        
        x_y_dim = 8  # Fixed based on preprocessing
        
        q_extract = x[:, :, 0:1]
        x_extract = x[:, :, 1:x_y_dim+1]
        y_extract = x[:, :, x_y_dim+1:2*x_y_dim+1]
        z_extract = x[:, :, 2*x_y_dim+1:2*x_y_dim+2]
        dt_extract = x[:, :, 2*x_y_dim+2:2*x_y_dim+3]
        
        if pred_mask is not None:
            if pred_mask.dim() == 2:
                pred_mask_expanded = pred_mask.unsqueeze(-1)  
            else:
                pred_mask_expanded = pred_mask
            
            # Zero out the MASKED positions (where pred_mask == True)
            mask_float = pred_mask_expanded.float()
            q_masked = q_extract * (1 - mask_float) 
            dt_masked = dt_extract * (1 - mask_float)
            
            # Process using the masked input values
            q_expanded = self.q_expander(q_masked)
            dt_expanded = self.dt_expander(dt_masked)
            
            # Add mask embedding ONLY to the masked positions
            mask_contribution = self.mask_embedding * mask_float
            q_expanded = q_expanded + mask_contribution
            dt_expanded = dt_expanded + mask_contribution
        else:
            # No masking case
            q_expanded = self.q_expander(q_extract)
            dt_expanded = self.dt_expander(dt_extract)
        
        x_embedded = self.x_encoder(x_extract)
        y_embedded = self.y_encoder(y_extract)
        z_embedded = self.z_encoder(z_extract).repeat(1, 1, self.embedding_dim)

        #if mask_ratio == 1.0:
        #    x_embedded = torch.zeros_like(x_embedded)
        #    y_embedded = torch.zeros_like(y_embedded)
        #    z_embedded = torch.zeros_like(z_embedded)
        
        combined_features = q_expanded + dt_expanded + x_embedded + y_embedded + z_embedded
        
        padding_mask = ~attention_mask
        transformer_output = self.transformer(
            combined_features,
            src_key_padding_mask=padding_mask
        )
        
        q_pred = self.q_decoder(transformer_output)
        dt_pred = self.dt_decoder(transformer_output)
        
        output = torch.zeros_like(x)
        output[:, :, 0:1] = q_pred
        output[:, :, 2*x_y_dim+2:2*x_y_dim+3] = dt_pred
        
        return output
        