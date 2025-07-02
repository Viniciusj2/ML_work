import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
import h5py
from Classifier_Dataset import ClassificationDataset, create_classification_dataloaders

class MAEBinaryClassifier(nn.Module):
    """
    Binary classifier that uses pre-trained MAE encoder features
    """
    def __init__(
        self,
        mae_model: nn.Module,
        freeze_encoder: bool = True,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
        pooling_strategy: str = "mean"  
    ):
        super().__init__()
        
        # Store the pre-trained MAE model
        self.mae_model = mae_model
        self.pooling_strategy = pooling_strategy
        
        # Freeze MAE encoder if specified
        if freeze_encoder:
            for param in self.mae_model.parameters():
                param.requires_grad = False
            print("MAE model parameters frozen")
        
        # Get encoder embedding dimension
        encoder_dim = mae_model.encoder_embedding_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, 1)  # Binary classification
        )
        
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def pool_features(self, encoder_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool encoder features into a single representation per event
        
        Args:
            encoder_output: [batch_size, seq_len, encoder_dim]
            attention_mask: [batch_size, seq_len]
        """
        if self.pooling_strategy == "mean":
            # mean pooling
            masked_output = encoder_output * attention_mask.unsqueeze(-1)
            pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
        elif self.pooling_strategy == "max":
            # max pooling
            masked_output = encoder_output.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            pooled = masked_output.max(dim=1)[0]
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def forward(self, event_tensor: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification
        
        Args:
            event_tensor: [batch_size, seq_len, input_dim]
            attention_mask: [batch_size, seq_len]
        """
        # Get encoder features from MAE model (without masking for classification)
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.mae_model.parameters())):
            # Extract encoder features
            batch_size, seq_len, _ = event_tensor.shape
            
            # Embed input tokens and add positional encoding
            token_embeddings = self.mae_model.encoder_embed(event_tensor)
            position_ids = torch.arange(seq_len, device=token_embeddings.device)
            token_embeddings = token_embeddings + self.mae_model.encoder_position_embeddings[position_ids]
            
            # Apply attention mask
            token_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
            
            # Run through encoder
            encoder_output = self.mae_model.encoder(
                token_embeddings,
                src_key_padding_mask=~attention_mask
            )
        
        # Pool features to get event-level representation
        pooled_features = self.pool_features(encoder_output, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return {
            "logits": logits,
            "features": pooled_features,
            "encoder_output": encoder_output
        }
