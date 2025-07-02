mport torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
import h5py
import os 
import wandb
from tqdm import tqdm
from datetime import datetime
from Classifier_Model import MAEBinaryClassifier
#from MAE_Transformer_Model import MAETransformerModel
from Transformer_Model_multi_residual import MAETransformerModel
from Classifier_Dataset import ClassificationDataset, create_classification_dataloaders


def train_classifier(
    classifier: MAEBinaryClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    project_name: str = "mae-binary-classifier",
    experiment_name: Optional[str] = None,
    config: Optional[dict] = None,
    save_dir: str = "checkpoints"
):
    """
    Training loop for the binary classifier with W&B logging and progress bars
    
    Args:
        classifier: The MAEBinaryClassifier model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        project_name: W&B project name
        experiment_name: W&B run name (optional)
        config: Additional config to log to W&B
        save_dir: Directory to save model checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize W&B
    if experiment_name is None:
        experiment_name = f"mae-classifier-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Prepare config for W&B
    wandb_config = {
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": train_loader.batch_size,
        "device": device,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "loss_function": "BCEWithLogitsLoss",
        "model_type": "MAEBinaryClassifier"
    }
    
    # Add any additional config
    if config:
        wandb_config.update(config)
    
    # Initialize wandb run
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=wandb_config,
        save_code=True
    )
    
    # Watch model for gradients and parameters
    wandb.watch(classifier, log="all", log_freq=100)
    
    classifier = classifier.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, f"best_{experiment_name}.pth")
    
    # Training metrics tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training progress bar
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} - Training", 
            leave=False,
            unit="batch"
        )
        
        for batch_idx, batch in enumerate(train_pbar):
            event_tensor = batch["event_tensor"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            outputs = classifier(event_tensor, attention_mask)
            logits = outputs["logits"].squeeze()
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update training progress bar with current metrics
            current_train_acc = train_correct / train_total
            current_train_loss = train_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Loss': f'{current_train_loss:.4f}',
                'Acc': f'{current_train_acc:.4f}'
            })
            
            # Log batch metrics every 50 batches
            if batch_idx % 50 == 0:
                wandb.log({
                    "batch_train_loss": loss.item(),
                    "batch_train_acc": (predictions == labels).float().mean().item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        train_pbar.close()
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Store predictions for confusion matrix
        all_predictions = []
        all_labels = []
        
        # Validation progress bar
        val_pbar = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} - Validation", 
            leave=False,
            unit="batch"
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                event_tensor = batch["event_tensor"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = classifier(event_tensor, attention_mask)
                logits = outputs["logits"].squeeze()
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                predictions = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                # Update validation progress bar with current metrics
                current_val_acc = val_correct / val_total
                current_val_loss = val_loss / (batch_idx + 1)
                val_pbar.set_postfix({
                    'Loss': f'{current_val_loss:.4f}',
                    'Acc': f'{current_val_acc:.4f}'
                })
                
                # Store for confusion matrix
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_pbar.close()
        
        # Calculate epoch metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(current_lr)
        
        # Log epoch metrics to W&B
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr,
            "best_val_acc": max(best_val_acc, val_acc)
        }
        
        # Add confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
            cm = wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_predictions,
                class_names=["Cosmic", "Marley"]
            )
            epoch_metrics["confusion_matrix"] = cm
        
        wandb.log(epoch_metrics)
        
        # Update main epoch progress bar with summary metrics
        epoch_pbar.set_postfix({
            'T_Loss': f'{avg_train_loss:.4f}',
            'T_Acc': f'{train_acc:.4f}',
            'V_Loss': f'{avg_val_loss:.4f}',
            'V_Acc': f'{val_acc:.4f}',
            'LR': f'{current_lr:.1e}'
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save model with metadata
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'config': wandb_config
            }
            
            torch.save(checkpoint, best_model_path)
            tqdm.write(f"  🎉 New best validation accuracy: {best_val_acc:.4f}")
            
            # Save model artifact to W&B
            model_artifact = wandb.Artifact(
                name=f"best-model-{experiment_name}",
                type="model",
                description=f"Best model checkpoint at epoch {epoch+1} with val_acc={best_val_acc:.4f}"
            )
            model_artifact.add_file(best_model_path)
            wandb.log_artifact(model_artifact)
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_dir, f"{experiment_name}_epoch_{epoch+1}.pth")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'config': wandb_config
            }
            torch.save(checkpoint, checkpoint_path)
            tqdm.write(f"  💾 Checkpoint saved at epoch {epoch+1}")
    
    epoch_pbar.close()
    
    # Log final training curves
    epochs_range = list(range(1, num_epochs + 1))
    
    # Create custom plots
    wandb.log({
        "training_curves": wandb.plot.line_series(
            xs=epochs_range,
            ys=[train_losses, val_losses],
            keys=["Train Loss", "Val Loss"],
            title="Training and Validation Loss",
            xname="Epoch"
        ),
        "accuracy_curves": wandb.plot.line_series(
            xs=epochs_range,
            ys=[train_accuracies, val_accuracies],
            keys=["Train Accuracy", "Val Accuracy"],
            title="Training and Validation Accuracy",
            xname="Epoch"
        ),
        "learning_rate_schedule": wandb.plot.line(
            table=wandb.Table(data=[[i, lr] for i, lr in enumerate(learning_rates, 1)], 
                             columns=["Epoch", "Learning Rate"]),
            x="Epoch", y="Learning Rate",
            title="Learning Rate Schedule"
        )
    })
    
    # Log final metrics summary
    wandb.summary.update({
        "final_train_acc": train_accuracies[-1],
        "final_val_acc": val_accuracies[-1],
        "best_val_acc": best_val_acc,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_epochs": num_epochs,
        "best_model_path": best_model_path
    })
    
    # Close W&B run
    wandb.finish()
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"💾 Best model saved to: {best_model_path}")
    print(f"📊 W&B run: {wandb.run.url if wandb.run else 'N/A'}")
    
    return classifier


def load_mae_model_from_checkpoint(checkpoint_path: str, model_config: dict) -> nn.Module:
    """
    Load MAE model from checkpoint that contains training metadata
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        model_config: Dictionary with model configuration parameters
    """
    # Create model with same config as training
    mae_model = MAETransformerModel(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    else:
        # Assume the entire checkpoint is the state dict
        model_state_dict = checkpoint
    
    # Load the state dict
    mae_model.load_state_dict(model_state_dict)
    
    print(f"Loaded MAE model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint was saved at epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return mae_model


# Example usage:
if __name__ == "__main__":
    # Configuration for your MAE model (must match training config)
    mae_config = {
        'input_dim': 19,
        'encoder_embedding_dim': 256,
        'decoder_embedding_dim': 128,
        'encoder_num_heads': 16,
        'decoder_num_heads': 8,
        'encoder_num_layers': 12,
        'decoder_num_layers': 6,
        'encoder_mlp_ratio': 16,
        'decoder_mlp_ratio': 8,
        'dropout': 0.1,
        'max_seq_length': 122
    }
    
    # Load pre-trained MAE model from checkpoint
    mae_model = load_mae_model_from_checkpoint(
        checkpoint_path="checkpoints_cluster/model_epoch_100.pth",
        model_config=mae_config
    )
    
    # Set to evaluation mode
    mae_model.eval()
    
    # Create binary classifier
    classifier = MAEBinaryClassifier(
        mae_model=mae_model,
        freeze_encoder=True,  # Keep MAE features frozen
        pooling_strategy="max"  # Try different pooling strategies
    )
    
    # Create dataloaders
    train_loader, val_loader = create_classification_dataloaders(
        cosmic_data_path="preprocessed_data/cosmic_events_processed.h5",
        marley_data_path="preprocessed_data/marley_events_processed.h5", 
        pos_encoding_path="preprocessed_data/position_encodings.npz",
        batch_size=32,
        max_waveforms_per_class=None
    )
    
    # Additional config for W&B logging
    experiment_config = {
        "mae_config": mae_config,
        "freeze_encoder": True,
        "pooling_strategy": "max",
        "max_waveforms_per_class": None,
        "dataset": "cosmic_vs_marley"
    }
    
    # Train the classifier with W&B logging
    trained_classifier = train_classifier(
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=1e-3,
        project_name="mae-binary-classifier",
        experiment_name="mae-classifier-v2_max",
        config=experiment_config,
        save_dir="checkpoints_max_cluster"
    )
