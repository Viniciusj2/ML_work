import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Tuple, Optional
import numpy as np
import h5py
import os 
import wandb
from tqdm import tqdm
from datetime import datetime
from Classifier_Model import MAEBinaryClassifier
from Transformer_Model_multi_residual import MAETransformerModel
from Classifier_Dataset import ClassificationDataset, create_classification_dataloaders


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def create_distributed_dataloaders(
    cosmic_data_path: str,
    marley_data_path: str,
    pos_encoding_path: str,
    batch_size: int,
    max_waveforms_per_class: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    train_split: float = 0.8
):
    """Create distributed dataloaders for multi-GPU training"""
    from Classifier_Dataset import ClassificationDataset
    
    # Create full dataset
    full_dataset = ClassificationDataset(
        cosmic_data_path=cosmic_data_path,
        marley_data_path=marley_data_path,
        pos_encoding_path=pos_encoding_path,
        max_waveforms_per_class=max_waveforms_per_class
    )
    
    # Split into train/val
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_sampler, val_sampler

def train_classifier_distributed(
    classifier: MAEBinaryClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: DistributedSampler,
    val_sampler: DistributedSampler,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    rank: int = 0,
    world_size: int = 1,
    project_name: str = "mae-binary-classifier",
    experiment_name: Optional[str] = None,
    config: Optional[dict] = None,
    save_dir: str = "checkpoints"
):
    """
    Training loop for the binary classifier with multi-GPU support and W&B logging
    """
    # Create save directory
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    # Only initialize W&B on rank 0
    if rank == 0:
        if experiment_name is None:
            experiment_name = f"mae-classifier-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Prepare config for W&B
        wandb_config = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_loader.batch_size,
            "world_size": world_size,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "loss_function": "BCEWithLogitsLoss",
            "model_type": "MAEBinaryClassifier",
            "distributed": world_size > 1
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
    
    # Move model to GPU and wrap with DDP
    device = torch.device(f"cuda:{rank}")
    classifier = classifier.to(device)
    
    if world_size > 1:
        classifier = DDP(classifier, device_ids=[rank], find_unused_parameters=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0.0
    if rank == 0:
        best_model_path = os.path.join(save_dir, f"best_{experiment_name}.pth")
    
    # Training metrics tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    # Main epoch progress bar (only on rank 0)
    if rank == 0:
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    else:
        epoch_pbar = range(num_epochs)
    
    for epoch in epoch_pbar:
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training progress bar (only on rank 0)
        if rank == 0:
            train_pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs} - Training", 
                leave=False,
                unit="batch"
            )
        else:
            train_pbar = train_loader
        
        for batch_idx, batch in enumerate(train_pbar):
            event_tensor = batch["event_tensor"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            outputs = classifier(event_tensor, attention_mask)
            logits = outputs["logits"].squeeze()
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.3).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update training progress bar with current metrics (only on rank 0)
            if rank == 0:
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
        
        if rank == 0:
            train_pbar.close()
        
        # Gather training metrics from all processes
        train_loss_tensor = torch.tensor(train_loss, device=device)
        train_correct_tensor = torch.tensor(train_correct, device=device)
        train_total_tensor = torch.tensor(train_total, device=device)
        
        if world_size > 1:
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        
        avg_train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)
        train_acc = train_correct_tensor.item() / train_total_tensor.item()
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Store predictions for confusion matrix (only on rank 0)
        if rank == 0:
            all_predictions = []
            all_labels = []
        
        # Validation progress bar (only on rank 0)
        if rank == 0:
            val_pbar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs} - Validation", 
                leave=False,
                unit="batch"
            )
        else:
            val_pbar = val_loader
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                event_tensor = batch["event_tensor"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                
                outputs = classifier(event_tensor, attention_mask)
                logits = outputs["logits"].squeeze()
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                predictions = (torch.sigmoid(logits) > 0.3).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                # Update validation progress bar with current metrics (only on rank 0)
                if rank == 0:
                    current_val_acc = val_correct / val_total
                    current_val_loss = val_loss / (batch_idx + 1)
                    val_pbar.set_postfix({
                        'Loss': f'{current_val_loss:.4f}',
                        'Acc': f'{current_val_acc:.4f}'
                    })
                    
                    # Store for confusion matrix
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        if rank == 0:
            val_pbar.close()
        
        # Gather validation metrics from all processes
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_correct_tensor = torch.tensor(val_correct, device=device)
        val_total_tensor = torch.tensor(val_total, device=device)
        
        if world_size > 1:
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
        
        avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
        val_acc = val_correct_tensor.item() / val_total_tensor.item()
        
        # Update learning rate (only on rank 0 to avoid conflicts)
        if rank == 0:
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
            if hasattr(epoch_pbar, 'set_postfix'):
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
                
                # Get the actual model from DDP wrapper
                model_to_save = classifier.module if hasattr(classifier, 'module') else classifier
                
                # Save model with metadata
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss,
                    'config': wandb_config if 'wandb_config' in locals() else {}
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
                model_to_save = classifier.module if hasattr(classifier, 'module') else classifier
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss,
                    'config': wandb_config if 'wandb_config' in locals() else {}
                }
                torch.save(checkpoint, checkpoint_path)
                tqdm.write(f"  💾 Checkpoint saved at epoch {epoch+1}")
        
        # Synchronize all processes
        if world_size > 1:
            dist.barrier()
    
    if rank == 0:
        if hasattr(epoch_pbar, 'close'):
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
    
    return classifier


def train_single_process(
    rank: int,
    world_size: int,
    mae_config: dict,
    mae_checkpoint_path: str,
    cosmic_data_path: str,
    marley_data_path: str,
    pos_encoding_path: str,
    training_config: dict
):
    """Single process training function for multiprocessing"""
    try:
        # Setup distributed training
        if world_size > 1:
            setup_distributed(rank, world_size)
        
        # Load pre-trained MAE model
        mae_model = load_mae_model_from_checkpoint(mae_checkpoint_path, mae_config)
        mae_model.eval()
        
        # Create binary classifier
        classifier = MAEBinaryClassifier(
            mae_model=mae_model,
            freeze_encoder=training_config.get("freeze_encoder", True),
            pooling_strategy=training_config.get("pooling_strategy", "max")
        )
        
        # Create distributed dataloaders
        train_loader, val_loader, train_sampler, val_sampler = create_distributed_dataloaders(
            cosmic_data_path=cosmic_data_path,
            marley_data_path=marley_data_path,
            pos_encoding_path=pos_encoding_path,
            batch_size=training_config.get("batch_size", 32),
            max_waveforms_per_class=training_config.get("max_waveforms_per_class"),
            rank=rank,
            world_size=world_size,
            num_workers=4
        )
        
        # Train the classifier
        trained_classifier = train_classifier_distributed(
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            num_epochs=training_config.get("num_epochs", 10),
            learning_rate=training_config.get("learning_rate", 1e-3),
            rank=rank,
            world_size=world_size,
            project_name=training_config.get("project_name", "mae-binary-classifier"),
            experiment_name=training_config.get("experiment_name"),
            config=training_config.get("experiment_config", {}),
            save_dir=training_config.get("save_dir", "checkpoints")
        )
        
    finally:
        # Clean up distributed training
        if world_size > 1:
            cleanup_distributed()


def load_mae_model_from_checkpoint(checkpoint_path: str, model_config: dict) -> nn.Module:
    """
    Load MAE model from checkpoint that contains training metadata
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


def main():
    """Main function to launch single-GPU or multi-GPU training"""
    # Check available GPUs
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    if world_size == 0:
        print("No GPUs available, using CPU training")
        world_size = 1
    
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
    
    # Training configuration
    training_config = {
        "num_epochs": 10,
        "learning_rate": 1e-3,
        "batch_size": 32,  # Per GPU batch size
        "freeze_encoder": True,
        "pooling_strategy": "max",
        "max_waveforms_per_class": None,
        "project_name": "mae-binary-classifier",
        "experiment_name": f"mae-classifier-v2_max_gpu{world_size}",
        "save_dir": "checkpoints_max_distributed",
        "experiment_config": {
            "mae_config": mae_config,
            "freeze_encoder": True,
            "pooling_strategy": "max",
            "max_waveforms_per_class": None,
            "dataset": "cosmic_vs_marley",
            "multi_gpu": world_size > 1
        }
    }
    
    # Data paths
    data_paths = {
        "mae_checkpoint_path": "checkpoints_resumed_50_100_LR_5e5/model_epoch_100.pth",
        "cosmic_data_path": "cosmic_events_processed.h5",
        "marley_data_path": "marley_events_processed.h5",
        "pos_encoding_path": "position_encodings.npz"
    }
    
    if world_size > 1:
        print(f"Launching distributed training on {world_size} GPUs")
        # Launch distributed training
        mp.spawn(
            train_single_process,
            args=(
                world_size,
                mae_config,
                data_paths["mae_checkpoint_path"],
                data_paths["cosmic_data_path"],
                data_paths["marley_data_path"],
                data_paths["pos_encoding_path"],
                training_config
            ),
            nprocs=world_size,
            join=True
        )
    else:
        print("Launching single GPU/CPU training")
        # Single GPU training
        train_single_process(
            rank=0,
            world_size=1,
            mae_config=mae_config,
            mae_checkpoint_path=data_paths["mae_checkpoint_path"],
            cosmic_data_path=data_paths["cosmic_data_path"],
            marley_data_path=data_paths["marley_data_path"],
            pos_encoding_path=data_paths["pos_encoding_path"],
            training_config=training_config
        )


if __name__ == "__main__":
    main()
