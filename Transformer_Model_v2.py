import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

class MAETransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 19,  
        spatial_dim: int = 17,  # 8 for x + 8 for y + 1 for z
        non_spatial_dim: int = 2,  # 1 for charge + 1 for delta time
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
        self.spatial_dim = spatial_dim
        self.non_spatial_dim = non_spatial_dim

        # Separate projections for spatial and non-spatial features
        self.spatial_proj = nn.Linear(spatial_dim, encoder_embedding_dim // 2)
        self.non_spatial_proj = nn.Linear(non_spatial_dim, encoder_embedding_dim // 2)
        
        # Position ID embeddings for decoder - NEW!
        self.position_embeddings = nn.Parameter(
            torch.zeros(max_seq_length, decoder_embedding_dim)
        )
        nn.init.normal_(self.position_embeddings, std=0.02)
        
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
        
        # Separate encoder-to-decoder projections
        self.spatial_encoder_to_decoder = nn.Linear(encoder_embedding_dim // 2, decoder_embedding_dim // 2, bias=False)
        self.non_spatial_encoder_to_decoder = nn.Linear(encoder_embedding_dim // 2, decoder_embedding_dim // 2, bias=False)
        
        # Separate decoder projections for visible tokens
        self.spatial_decoder_proj = nn.Linear(spatial_dim, decoder_embedding_dim // 2)
        self.non_spatial_decoder_proj = nn.Linear(non_spatial_dim, decoder_embedding_dim // 2)
        
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
        
        # Context projection - NEW!
        self.context_proj = nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, event_tensor, attention_mask, pred_mask=None):
        """ Assumes `event_tensor` already includes positional information """
        
        batch_size, seq_len, total_dim = event_tensor.shape
        
        # Split input into spatial and non-spatial components
        # Assuming: [charge, x_pos_enc(8), y_pos_enc(8), z_pos_enc(1), delta_time]
        non_spatial_features = torch.cat([
            event_tensor[:, :, 0:1],    # Charge
            event_tensor[:, :, -1:],    # Delta time
        ], dim=-1)
        
        spatial_features = event_tensor[:, :, 1:self.spatial_dim+1]  # x, y, z position encodings
        
        # --- Encoder Phase ---
        # Project spatial and non-spatial features separately
        spatial_embeddings = self.spatial_proj(spatial_features)
        non_spatial_embeddings = self.non_spatial_proj(non_spatial_features)
        
        # Concatenate embeddings
        token_embeddings = torch.cat([spatial_embeddings, non_spatial_embeddings], dim=-1)
        
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
        
        # Initialize padding mask for encoder
        # In PyTorch transformer, padding_mask=True means the position is ignored
        encoder_padding_mask = torch.ones(batch_size, max_visible, dtype=torch.bool, 
                                        device=event_tensor.device)
        
        if batch_size > 0 and max_visible > 0:
            # Debug output - limit to first few batches to avoid flooding
            if batch_size < 3:
                print(f"\nInitial encoder padding mask (all True = all masked):")
                print(f"Shape: {encoder_padding_mask.shape}")
        
        seq_to_encoder_idx = torch.full((batch_size, seq_len), -1, 
                                    dtype=torch.long, device=event_tensor.device)
        
        # Store spatial features for each position - NEW!
        pmt_position_features = spatial_features.clone()
        
        for b in range(batch_size):
            visible_pos = visible_mask[b].nonzero(as_tuple=True)[0]
            count = len(visible_pos)
            if count > 0:
                visible_embeddings[b, :count] = token_embeddings[b, visible_pos]
                seq_to_encoder_idx[b, visible_pos] = torch.arange(count, device=event_tensor.device)
                # Set positions corresponding to actual tokens to False (not masked)
                encoder_padding_mask[b, :count] = False
                
                # Debug output - limit to first few batches to avoid flooding
                if b < 3:
                    print(f"\nBatch {b}:")
                    print(f"Visible positions count: {count}/{seq_len}")
                    print(f"Encoder mask - False (valid) count: {(~encoder_padding_mask[b]).sum().item()}")
                    print(f"Encoder mask - True (masked) count: {encoder_padding_mask[b].sum().item()}")
            else:
                if b < 3:
                    print(f"\nBatch {b}: No visible positions")
                
        # More verbose debugging only for very small batches
        if batch_size <= 2 and max_visible <= 64:
            print("\nFinal encoder padding mask:")
            for b in range(batch_size):
                print(f"Batch {b}: {encoder_padding_mask[b].cpu().numpy()}")
            
            print(f"\nTotal across all batches:")
            print(f"Valid positions (False): {(~encoder_padding_mask).sum().item()}")
            print(f"Masked positions (True): {encoder_padding_mask.sum().item()}")
        
        # Encode visible tokens 
        encoder_output = self.encoder(
            visible_embeddings,
            src_key_padding_mask=encoder_padding_mask  # True = position is ignored
        )
        
        # Create a global context vector from the encoder output - NEW!
        # Average pool over the visible tokens for each batch
        global_context = []
        for b in range(batch_size):
            valid_count = (~encoder_padding_mask[b]).sum().item()
            if valid_count > 0:
                # Average pool over valid tokens only
                batch_context = encoder_output[b, :valid_count].mean(dim=0, keepdim=True)
                global_context.append(batch_context)
            else:
                # Fallback if no valid tokens
                global_context.append(torch.zeros(1, self.encoder_embedding_dim, device=encoder_output.device))
                
        global_context = torch.cat(global_context, dim=0).unsqueeze(1)  # [batch_size, 1, encoder_dim]
        
        # --- Decoder Phase ---
        # Split encoder output back into spatial and non-spatial components
        encoder_spatial = encoder_output[:, :, :self.encoder_embedding_dim // 2]
        encoder_non_spatial = encoder_output[:, :, self.encoder_embedding_dim // 2:]

        # Project each component separately to decoder dim
        decoder_spatial = self.spatial_encoder_to_decoder(encoder_spatial)
        decoder_non_spatial = self.non_spatial_encoder_to_decoder(encoder_non_spatial)

        # Combine for decoder visible tokens
        decoder_visible = torch.cat([decoder_spatial, decoder_non_spatial], dim=-1)

        # Initialize decoder input with mask tokens for all positions
        decoder_input = self.mask_token.expand(batch_size, seq_len, -1).clone()

        # Project global context to decoder dimension
        global_context_proj = self.context_proj(global_context)  # [batch_size, 1, decoder_dim]
        
        # Fill in visible tokens with their encoded representations
        for b in range(batch_size):
            visible_pos = (seq_to_encoder_idx[b] >= 0).nonzero(as_tuple=True)[0]
            if len(visible_pos) > 0:
                decoder_input[b, visible_pos] = decoder_visible[b, seq_to_encoder_idx[b, visible_pos]]
        
        # Add position embeddings to decoder input - NEW!
        position_ids = torch.arange(seq_len, device=decoder_input.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings[position_ids]
        decoder_input = decoder_input + position_embeddings
        
        # Add the global context to all positions - NEW!
        # This helps with distribution matching across the event
        decoder_input = decoder_input + global_context_proj.expand(-1, seq_len, -1)
        
        # For decoder, ~attention_mask means positions to ignore
        # This is because attention_mask is True for valid positions in your implementation
        decoder_output = self.decoder(
            decoder_input,
            src_key_padding_mask=~attention_mask  # True = position is ignored
        )
        
        # Spatial feature conditioning for predictions - NEW!
        # Make predictions spatially aware based on PMT positions
        pmt_position_decoder = self.spatial_proj(pmt_position_features)
        
        # Combine decoder output with spatial conditioning for prediction
        prediction_input = decoder_output + pmt_position_decoder[:, :, :decoder_output.shape[-1]//2].repeat(1, 1, 2)
        
        # Predictions 
        q_pred = self.q_pred_head(prediction_input)
        dt_pred = self.dt_pred_head(prediction_input)
        
        return {
            "output": torch.cat([q_pred, dt_pred], dim=-1),
            "attention": None
        }
    

    def compute_loss(self, pred_values, original_values, pred_mask, attention_mask):
        """Compute loss for masked prediction with structure-aware components"""
        valid_mask = pred_mask & attention_mask
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_values.device), {
                "q_loss": torch.tensor(0.0, device=pred_values.device),
                "dt_loss": torch.tensor(0.0, device=pred_values.device),
                "gradient_loss": torch.tensor(0.0, device=pred_values.device),
                "total_loss": torch.tensor(0.0, device=pred_values.device),
                "valid_positions": 0
            }
        
        valid_indices = valid_mask.nonzero(as_tuple=True)
        pred_q = pred_values[valid_indices[0], valid_indices[1], 0]
        pred_dt = pred_values[valid_indices[0], valid_indices[1], 1]
        
        true_q = original_values[valid_indices[0], valid_indices[1], 0]
        true_dt = original_values[valid_indices[0], valid_indices[1], 1]
        
        # Basic reconstruction losses
        # Use Huber loss for q to be more robust to outliers
        q_loss = F.huber_loss(pred_q, true_q, delta=0.1)
        dt_loss = F.mse_loss(pred_dt, true_dt)
        
        # Add structure-aware components
        gradient_loss = torch.tensor(0.0, device=pred_values.device)
        
        # Process each sequence in the batch separately for gradient calculation
        for b in range(pred_values.shape[0]):
            # Get sorted indices for this batch
            batch_mask = valid_mask[b]
            if batch_mask.sum() <= 1:
                continue  # Skip if not enough points for gradient
                
            seq_indices = batch_mask.nonzero(as_tuple=True)[0]
            seq_indices_sorted, sort_order = torch.sort(seq_indices)
            
            if len(seq_indices_sorted) <= 1:
                continue
                
            # Extract ordered predictions and targets for this sequence
            b_pred_q = pred_values[b, seq_indices_sorted, 0]
            b_true_q = original_values[b, seq_indices_sorted, 0]
            b_pred_dt = pred_values[b, seq_indices_sorted, 1]
            b_true_dt = original_values[b, seq_indices_sorted, 1]
            
            # Calculate gradients (differences between adjacent points)
            pred_q_grad = b_pred_q[1:] - b_pred_q[:-1]
            true_q_grad = b_true_q[1:] - b_true_q[:-1]
            pred_dt_grad = b_pred_dt[1:] - b_pred_dt[:-1]
            true_dt_grad = b_true_dt[1:] - b_true_dt[:-1]
            
            # Add gradient losses
            seq_gradient_loss = (
                F.mse_loss(pred_q_grad, true_q_grad) + 
                F.mse_loss(pred_dt_grad, true_dt_grad)
            )
            gradient_loss += seq_gradient_loss
        
        # Normalize gradient loss by batch size
        if pred_values.shape[0] > 0:
            gradient_loss = gradient_loss / pred_values.shape[0]
        
        # Add variability penalty to discourage predicting just mean values
        q_var_loss = torch.abs(pred_q.var() - true_q.var())
        dt_var_loss = torch.abs(pred_dt.var() - true_dt.var())
        variance_loss = q_var_loss + dt_var_loss
        
        # Weight the loss components
        total_loss = q_loss + dt_loss + 0.5 * gradient_loss + 0.2 * variance_loss
        
        return total_loss, {
            "q_loss": q_loss.detach(),
            "dt_loss": dt_loss.detach(),
            "gradient_loss": gradient_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "total_loss": total_loss.detach(),
            "valid_positions": valid_mask.sum().item()
        }
