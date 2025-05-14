import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import h5py

class MAETransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 19,  
        encoder_embedding_dim: int = 256,
        decoder_embedding_dim: int = 128,
        encoder_num_heads: int = 8,
        decoder_num_heads: int = 4,
        encoder_num_layers: int = 6,
        decoder_num_layers: int = 2,
        encoder_mlp_ratio: int = 4,
        decoder_mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 122  
    ):
        super(MAETransformerModel, self).__init__()
        
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.max_seq_length = max_seq_length

        # Encoder projection
        self.encoder_input_proj = nn.Linear(input_dim, encoder_embedding_dim)
        
        # Encoder transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_embedding_dim,
            nhead=encoder_num_heads,
            dim_feedforward=encoder_mlp_ratio * encoder_embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num_layers)
        
        # Decoder components
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        self.encoder_to_decoder = nn.Linear(encoder_embedding_dim, decoder_embedding_dim, bias=False)
        
        # Decoder projection (for visible tokens) 
        self.decoder_input_proj = nn.Linear(input_dim, decoder_embedding_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embedding_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_mlp_ratio * decoder_embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_num_layers)
        
        # Prediction heads
        self.q_pred_head = nn.Linear(decoder_embedding_dim, 1)
        self.dt_pred_head = nn.Linear(decoder_embedding_dim, 1)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, event_tensor, attention_mask, pred_mask=None):
        """ Assumes `event_tensor` already includes positional information """
        
        batch_size, seq_len, _ = event_tensor.shape
        
        # --- Encoder Phase ---
        # Project all features (visible + masked)
        token_embeddings = self.encoder_input_proj(event_tensor)  
        
        # Create visible tokens mask
        visible_mask = attention_mask.clone()
        if pred_mask is not None:
            visible_mask = visible_mask & (~pred_mask)
        
        # Process only visible tokens through encoder 
        visible_counts = visible_mask.sum(dim=1)
        max_visible = visible_counts.max().item()
        
        if max_visible == 0:
            return {
                "output": torch.zeros(batch_size, seq_len, 2, device=event_tensor.device),
                "attention": None
            }
        
        # Extract visible tokens 
        visible_embeddings = torch.zeros(
            batch_size, max_visible, self.encoder_embedding_dim,
            device=event_tensor.device
        )
        encoder_padding_mask = torch.ones(batch_size, max_visible, dtype=torch.bool, 
                                        device=event_tensor.device)
        
        seq_to_encoder_idx = torch.full((batch_size, seq_len), -1, 
                                    dtype=torch.long, device=event_tensor.device)
        
        for b in range(batch_size):
            visible_pos = visible_mask[b].nonzero(as_tuple=True)[0]
            count = len(visible_pos)
            if count > 0:
                visible_embeddings[b, :count] = token_embeddings[b, visible_pos]
                encoder_padding_mask[b, :count] = False
                seq_to_encoder_idx[b, visible_pos] = torch.arange(count, device=event_tensor.device)
        
        # Encode visible tokens 
        encoder_output = self.encoder(
            visible_embeddings,
            src_key_padding_mask=encoder_padding_mask
        )
        
        # --- Decoder Phase ---
        # Project encoder output to decoder dim 
        decoder_visible = self.encoder_to_decoder(encoder_output)
        
        # Initialize decoder input with mask tokens 
        decoder_input = self.mask_token.expand(batch_size, seq_len, -1).clone()
        
        # Project original inputs for visible tokens 
        original_visible_proj = self.decoder_input_proj(event_tensor)
        
        # Insert encoded visible tokens 
        for b in range(batch_size):
            valid_pos = (seq_to_encoder_idx[b] >= 0).nonzero(as_tuple=True)[0]
            if len(valid_pos) > 0:
                decoder_input[b, valid_pos] = decoder_visible[b, seq_to_encoder_idx[b, valid_pos]]
    
        # Decode full sequence 
        decoder_output = self.decoder(
            decoder_input,
            src_key_padding_mask=~attention_mask
        )
        
        # Predictions 
        q_pred = self.q_pred_head(decoder_output)
        dt_pred = self.dt_pred_head(decoder_output)
        
        return {
            "output": torch.cat([q_pred, dt_pred], dim=-1),
            "attention": None
        }

    def compute_loss(self, pred_values, original_values, pred_mask, attention_mask):
        """Identical to previous implementation"""
        valid_mask = pred_mask & attention_mask
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_values.device), {
                "q_loss": torch.tensor(0.0, device=pred_values.device),
                "dt_loss": torch.tensor(0.0, device=pred_values.device),
                "total_loss": torch.tensor(0.0, device=pred_values.device),
                "valid_positions": 0
            }
        
        valid_indices = valid_mask.nonzero(as_tuple=True)
        pred_q = pred_values[valid_indices[0], valid_indices[1], 0]
        pred_dt = pred_values[valid_indices[0], valid_indices[1], 1]
        
        true_q = original_values[valid_indices[0], valid_indices[1], 0]
        true_dt = original_values[valid_indices[0], valid_indices[1], 1]
        
        q_loss = F.mse_loss(pred_q, true_q)
        dt_loss = F.mse_loss(pred_dt, true_dt)
        total_loss = q_loss + dt_loss
        
        return total_loss, {
            "q_loss": q_loss.detach(),
            "dt_loss": dt_loss.detach(),
            "total_loss": total_loss.detach(),
            "valid_positions": valid_mask.sum().item()
        }