import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

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
        super().__init__()
        
        # Store dimensions for validation
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.max_seq_length = max_seq_length
        
        # Encoder components
        self.encoder_embed = nn.Sequential(
            nn.Linear(input_dim, encoder_embedding_dim),
            nn.LayerNorm(encoder_embedding_dim)
        )
        
        # Encoder position embeddings (needed to distinguish PMT positions)
        self.encoder_position_embeddings = nn.Parameter(
            torch.zeros(max_seq_length, encoder_embedding_dim)
        )
        nn.init.normal_(self.encoder_position_embeddings, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_embedding_dim,
            nhead=encoder_num_heads,
            dim_feedforward=encoder_mlp_ratio * encoder_embedding_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num_layers)
        
        # Decoder components
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Position embeddings for decoder (needed to distinguish PMT positions)
        self.decoder_position_embeddings = nn.Parameter(
            torch.zeros(max_seq_length, decoder_embedding_dim)
        )
        nn.init.normal_(self.decoder_position_embeddings, std=0.02)
        
        # Encoder to decoder projection with proper residual handling
        self.encoder_to_decoder = nn.Sequential(
            nn.Linear(encoder_embedding_dim, decoder_embedding_dim),
            nn.LayerNorm(decoder_embedding_dim)
        )
        
        # Decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embedding_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_mlp_ratio * decoder_embedding_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_num_layers)
        
        # Improved prediction heads with residual connections
        self.charge_head = nn.Sequential(
            nn.Linear(decoder_embedding_dim, decoder_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(decoder_embedding_dim, decoder_embedding_dim//2),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),  # Reduced dropout for final layer
            nn.Linear(decoder_embedding_dim//2, 1)
        )
        
        self.time_head = nn.Sequential(
            nn.Linear(decoder_embedding_dim, decoder_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(decoder_embedding_dim, decoder_embedding_dim//2),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),  # Reduced dropout for final layer
            nn.Linear(decoder_embedding_dim//2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use different initialization for different layer types
                if module.out_features == 1:  # Output layers
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, event_tensor, attention_mask, pred_mask=None):
        batch_size, seq_len, _ = event_tensor.shape
        
        # Validate sequence length
        if seq_len > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_length {self.max_seq_length}")
        
        # --- Encoder Phase ---
        visible_mask = attention_mask if pred_mask is None else attention_mask & (~pred_mask)
        
        # Embed input tokens and add positional encoding for PMT position identity
        token_embeddings = self.encoder_embed(event_tensor)
        
        # Add sequence positional embeddings (essential for distinguishing PMT positions)
        position_ids = torch.arange(seq_len, device=token_embeddings.device)
        token_embeddings = token_embeddings + self.encoder_position_embeddings[position_ids]
        
        # Apply visibility mask
        token_embeddings = token_embeddings * visible_mask.unsqueeze(-1)
        
        encoder_output = self.encoder(
            token_embeddings,
            src_key_padding_mask=~visible_mask
        )
        
        # --- Decoder Phase ---
        # Project encoder output to decoder dimension
        projected_encoder = self.encoder_to_decoder(encoder_output)
        
        # Initialize decoder input with mask tokens for all positions
        decoder_input = self.mask_token.expand(batch_size, seq_len, -1).clone()
        
        # For visible positions, use projected encoder outputs
        if pred_mask is not None:
            decoder_input = torch.where(
                visible_mask.unsqueeze(-1),
                projected_encoder,
                decoder_input
            )
        else:
            # If no prediction mask, use all encoder outputs
            decoder_input = projected_encoder
        
        # Add decoder positional embeddings (essential for PMT position identity)
        decoder_position_ids = torch.arange(seq_len, device=decoder_input.device)
        decoder_input = decoder_input + self.decoder_position_embeddings[decoder_position_ids]
        
        # Decode with proper attention mask
        decoder_output = self.decoder(
            decoder_input,
            src_key_padding_mask=~attention_mask
        )
        
        # Apply prediction heads
        charge_pred = self.charge_head(decoder_output)
        time_pred = self.time_head(decoder_output)
        predictions = torch.cat([charge_pred, time_pred], dim=-1)
        
        return {"output": predictions, "attention": None}
    
    def compute_loss(self, pred_values, original_values, pred_mask, attention_mask):
        """Same as before"""
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
        
        q_l1_loss = F.l1_loss(pred_q, true_q)
        dt_l1_loss = F.l1_loss(pred_dt, true_dt)
        
        q_l2_loss = F.mse_loss(pred_q, true_q)
        dt_l2_loss = F.mse_loss(pred_dt, true_dt)
        
        q_loss = q_l1_loss + q_l2_loss
        dt_loss = dt_l1_loss + dt_l2_loss
        total_loss = q_loss #+ dt_loss
        
        return total_loss, {
            "q_loss": q_loss.detach(),
            "dt_loss": dt_loss.detach(),
            "total_loss": total_loss.detach(),
            "valid_positions": valid_mask.sum().item()
        }
