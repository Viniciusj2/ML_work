import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from Transformer_Model import ModifiedTransformerModel
from Transformer_Dataset import PreprocessedPMTDataset, create_dataloader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
from collections import defaultdict
import wandb
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a self-supervised transformer model')
    
    # Dataset parameters
    parser.add_argument('--cosmic_file', type=str, required=False, default='preprocessed_data/cosmic_events_processed.h5', help='Path to preprocessed cosmic data file')
    parser.add_argument('--marley_file', type=str, required=False, default='preprocessed_data/marley_events_processed.h5', help='Path to preprocessed marley data file')
    parser.add_argument('--pos_encoding_file', type=str, required=False, default='preprocessed_data/position_encodings.npz', help='Path to position encodings file')
    parser.add_argument('--max_waveforms', type=int, default=None, help='Maximum number of waveforms to use')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,help='Number of gradient accumulation steps')
    parser.add_argument('--mixed_precision', type=lambda x: (str(x).lower() == 'true'), default=True,help='Use mixed precision training')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='30_masks_10_epochs',help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='transformer_pre-training',help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default='viniciusj_silva-tufts-university', help='Weights & Biases entity name')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default=None,help='Device to use (cuda or cpu, default: auto-detect)')
    parser.add_argument('--num_workers', type=int, default=2,help='Number of workers for dataloaders')
    
    return parser.parse_args()

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = None,
    checkpoint_dir: str = "30_masks_100_epochs",
    wandb_project: str = "transformer_pre-training",
    wandb_entity: str = "viniciusj_silva-tufts-university",
    patience: int = 100,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = True
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up CUDA events for precise GPU timing
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        def cuda_time_op(op_func):
            start_event.record()
            result = op_func()
            end_event.record()
            torch.cuda.synchronize()
            return result, start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    else:
        def cuda_time_op(op_func):
            start = time.time()
            result = op_func()
            end = time.time()
            return result, end - start
    
    # Initialize wandb
    wandb.login(key="*******************************")
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": next(iter(train_loader))[0].shape[0],
            "embedding_dim": model.embedding_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mixed_precision": mixed_precision
        }
    )
    # Watch model to track parameter updates and gradients
    wandb.watch(model, log="all", log_freq=100)
    
    # Balanced weights for q and dt losses
    q_weight = 1.0
    dt_weight = 1.0
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Things to Help with Training Time
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    # Tracking Timing 
    timing_stats = defaultdict(float)
    
    # Warm Up Dataloader to help with run time
    print("Warming up dataloader...")
    warmup_start = time.time()
    for _ in range(3):
        for batch in train_loader:
            _ = [t.to(device) for t in batch]
            break
    warmup_time = time.time() - warmup_start
    print(f"Dataloader warmup completed in {warmup_time:.2f} seconds")
    
    print("Benchmarking dataloader...")
    dataloader_bench_start = time.time()
    batch_times = []
    for i, batch in enumerate(train_loader):
        batch_start = time.time()
        _ = [t.to(device) for t in batch]
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        if i >= 10:  
            break
    dataloader_bench_time = time.time() - dataloader_bench_start
    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"Average batch loading time: {avg_batch_time:.4f} seconds")
    print(f"Total benchmark time: {dataloader_bench_time:.2f} seconds")
    
    x_y_dim = 8
    dt_index = x_y_dim*2+2

    for epoch in range(num_epochs):
        # Reset epoch timing stats
        epoch_timing_stats = {
            'train_data_loading': 0.0,
            'train_transfer_to_device': 0.0,
            'train_forward_pass': 0.0,
            'train_backward_pass': 0.0,
            'train_optimizer_step': 0.0,
            'train_misc_operations': 0.0,
            'validation_data_loading': 0.0,
            'validation_transfer_to_device': 0.0,
            'validation_forward_pass': 0.0,
            'validation_misc_operations': 0.0,
            'epoch_overhead': 0.0,
            'dataloader_overhead': 0.0
        }
        
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0
        train_q_loss = 0
        train_dt_loss = 0
        total_samples = 0
        optimizer.zero_grad()  
        
        # Training phase
        train_start_time = time.time()
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            batch_idx = 0
            dataloader_start = time.time()
            
            for batch in train_loader:
                # Record time between batches (dataloader overhead)
                dataloader_end = time.time()
                dataloader_time = dataloader_end - dataloader_start
                epoch_timing_stats['dataloader_overhead'] += dataloader_time
                
                # Batch timing
                batch_start = time.time()
                
              
                transfer_start = time.time()
                batch = [t.to(device) for t in batch]
                tensor, attention_mask, pred_mask = batch
                batch_size = tensor.size(0)
                total_samples += batch_size
                
                # Extract ground truth values
                q_truth = tensor[:, :, 0:1]
                dt_truth = tensor[:, :, x_y_dim*2+2:x_y_dim*2+3]
                transfer_end = time.time()
                epoch_timing_stats['train_transfer_to_device'] += transfer_end - transfer_start

                def forward_op():
                        with torch.cuda.amp.autocast(enabled=mixed_precision):
                            x_y_dim = 8  # Fixed based on preprocessing
                            
                            # Pass pred_mask to the model
                            output = model(tensor, attention_mask, pred_mask)
                            
                            # Extract predictions from output tensor at the correct indices
                            q_pred = output[:, :, 0:1]  
                            dt_pred = output[:, :, 2*x_y_dim+2:2*x_y_dim+3] 
                            
                            # Extract ground truth values at the same indices
                            q_truth = tensor[:, :, 0:1]  
                            dt_truth = tensor[:, :, 2*x_y_dim+2:2*x_y_dim+3]  
                            
                            # Calculate raw per-element losses
                            q_loss_raw = criterion(q_pred, q_truth)  
                            dt_loss_raw = criterion(dt_pred, dt_truth)  
                            
                            # Apply prediction mask - only calculate loss for masked tokens
                            if pred_mask.dim() == 2:
                                pred_mask_expanded = pred_mask.unsqueeze(-1)  
                            else:
                                pred_mask_expanded = pred_mask
                                
                            masked_q_loss = q_loss_raw * pred_mask_expanded
                            masked_dt_loss = dt_loss_raw * pred_mask_expanded
                            
                            # Calculate total number of masked tokens for normalization
                            mask_sum = pred_mask_expanded.sum() + 1e-8  # Add small epsilon to avoid division by zero
                            
                            # Sum losses and normalize 
                            q_loss = masked_q_loss.sum() / mask_sum
                            dt_loss = masked_dt_loss.sum() / mask_sum
                            
                            # Combine losses with weights
                            loss = q_weight * q_loss + dt_weight * dt_loss
                            
                            # Scale loss for gradient accumulation
                            scaled_loss = loss / gradient_accumulation_steps
                            
                            return scaled_loss, loss, q_loss, dt_loss
                        
                (scaled_loss, loss, q_loss, dt_loss), forward_time = cuda_time_op(forward_op)
                epoch_timing_stats['train_forward_pass'] += forward_time
                
                # Backward pass timing
                def backward_op():
                    if mixed_precision:
                        return scaler.scale(scaled_loss).backward()
                    else:
                        return scaled_loss.backward()
                
                _, backward_time = cuda_time_op(backward_op)
                epoch_timing_stats['train_backward_pass'] += backward_time
                
                # Store batch losses
                batch_loss = loss.item()
                batch_q_loss = q_loss.item()
                batch_dt_loss = dt_loss.item()
                
                # Accumulate total losses
                train_loss += batch_loss * batch_size
                train_q_loss += batch_q_loss * batch_size
                train_dt_loss += batch_dt_loss * batch_size
                
                # Optimizer Timing
                optimizer_time = 0
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                    def optimizer_op():
                        if mixed_precision:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                    
                    _, optimizer_time = cuda_time_op(optimizer_op)
                
                epoch_timing_stats['train_optimizer_step'] += optimizer_time
                
                #  progress bar
                pbar_start = time.time()
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'q_loss': f'{batch_q_loss:.4f}',
                    'dt_loss': f'{batch_dt_loss:.4f}',
                    'data_time': f'{dataloader_time:.3f}',
                    'compute_time': f'{forward_time + backward_time + optimizer_time:.3f}'
                })
                pbar_end = time.time()
                epoch_timing_stats['train_misc_operations'] += pbar_end - pbar_start
                
                #  batch end time
                batch_end = time.time()
                batch_total_time = batch_end - batch_start
                
                # unaccounted batch time
                accounted_batch_time = (
                    (transfer_end - transfer_start) + 
                    forward_time + 
                    backward_time + 
                    optimizer_time + 
                    (pbar_end - pbar_start)
                )
                unaccounted_batch_time = batch_total_time - accounted_batch_time
                
                if unaccounted_batch_time > 0.01:  
                    print(f"Batch {batch_idx} unaccounted time: {unaccounted_batch_time:.4f}s")
                
                batch_idx += 1
                # Start timing for next batch's dataloader
                dataloader_start = time.time()
        
        # Calculate total training time
        train_end_time = time.time()
        total_train_time = train_end_time - train_start_time
        
        # Normalize epoch-level losses
        train_loss /= total_samples if total_samples > 0 else 1
        train_q_loss /= total_samples if total_samples > 0 else 1
        train_dt_loss /= total_samples if total_samples > 0 else 1
        
        validation_start_time = time.time()
        model.eval()
        val_loss = 0
        val_q_loss = 0
        val_dt_loss = 0
        val_samples = 0
        
        val_dataloader_start = time.time()
        
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                val_dataloader_end = time.time()
                val_dataloader_time = val_dataloader_end - val_dataloader_start
                epoch_timing_stats['validation_data_loading'] += val_dataloader_time
                
                # Transfer to device
                val_transfer_start = time.time()
                val_batch = [t.to(device) for t in val_batch]
                tensor, attention_mask, pred_mask = val_batch
                batch_size = tensor.size(0)
                val_samples += batch_size
                
                # Extract ground truth 
                q_truth = tensor[:, :, 0:1]
                dt_truth = tensor[:, :, model.embedding_dim:model.embedding_dim+1]
                val_transfer_end = time.time()
                epoch_timing_stats['validation_transfer_to_device'] += val_transfer_end - val_transfer_start
                
                #Same idea as forward_op 
                def val_forward_op():
                    with torch.cuda.amp.autocast(enabled=mixed_precision):
                        output = model(tensor, attention_mask, pred_mask)
                        
                   
                        q_pred = output[:, :, 0:1]
                        dt_pred = output[:, :, 2*x_y_dim+2:2*x_y_dim+3]
                        
                        q_truth = tensor[:, :, 0:1]
                        dt_truth = tensor[:, :, 2*x_y_dim+2:2*x_y_dim+3]
                        
                        q_loss_raw = criterion(q_pred, q_truth)  
                        dt_loss_raw = criterion(dt_pred, dt_truth)  
                        
                        if pred_mask.dim() == 2:
                            pred_mask_expanded = pred_mask.unsqueeze(-1)  
                        else:
                            pred_mask_expanded = pred_mask
                            
                        masked_q_loss = q_loss_raw * pred_mask_expanded
                        masked_dt_loss = dt_loss_raw * pred_mask_expanded
                        
                        
                        mask_sum = pred_mask_expanded.sum() + 1e-8  # Add small epsilon to avoid division by zero
                        
                        # Sum losses and normalize by number of masked tokens
                        q_loss = masked_q_loss.sum() / mask_sum
                        dt_loss = masked_dt_loss.sum() / mask_sum
                        
                        # Combine losses with weights
                        loss = q_weight * q_loss + dt_weight * dt_loss
                        
                        return loss, q_loss, dt_loss
                    
                (loss, q_loss, dt_loss), val_forward_time = cuda_time_op(val_forward_op)
                epoch_timing_stats['validation_forward_pass'] += val_forward_time
                
                # Misc validation operations
                val_misc_start = time.time()
                val_loss += loss.item() * batch_size
                val_q_loss += q_loss.item() * batch_size
                val_dt_loss += dt_loss.item() * batch_size
                val_misc_end = time.time()
                epoch_timing_stats['validation_misc_operations'] += val_misc_end - val_misc_start
                
                # Start timing for next batch's dataloader
                val_dataloader_start = time.time()
        
        # Normalize validation losses
        val_loss /= val_samples if val_samples > 0 else 1
        val_q_loss /= val_samples if val_samples > 0 else 1
        val_dt_loss /= val_samples if val_samples > 0 else 1
        
        validation_end_time = time.time()
        total_validation_time = validation_end_time - validation_start_time
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Calculate remaining epoch overhead
        epoch_end_time = time.time()
        total_epoch_time = epoch_end_time - epoch_start_time
        
        total_accounted_time = total_train_time + total_validation_time
        epoch_overhead = total_epoch_time - total_accounted_time
        epoch_timing_stats['epoch_overhead'] = max(0, epoch_overhead)
        
        # Update timing
        for key, value in epoch_timing_stats.items():
            timing_stats[key] += value
        
        #  unaccounted training and validation time
        train_accounted_time = (
            epoch_timing_stats['train_transfer_to_device'] +
            epoch_timing_stats['train_forward_pass'] +
            epoch_timing_stats['train_backward_pass'] +
            epoch_timing_stats['train_optimizer_step'] +
            epoch_timing_stats['train_misc_operations']
        )
        train_unaccounted = total_train_time - train_accounted_time
        
        val_accounted_time = (
            epoch_timing_stats['validation_transfer_to_device'] +
            epoch_timing_stats['validation_forward_pass'] +
            epoch_timing_stats['validation_misc_operations']
        )
        val_unaccounted = total_validation_time - val_accounted_time
        
        # Log epoch-level metrics to wandb
        # Log loss metrics
        wandb_metrics = {
            'epoch/train_loss': train_loss,
            'epoch/val_loss': val_loss,
            'epoch/train_q_loss': train_q_loss,
            'epoch/val_q_loss': val_q_loss,
            'epoch/train_dt_loss': train_dt_loss,
            'epoch/val_dt_loss': val_dt_loss,
            'epoch/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch/number': epoch
        }
        
        # Add detailed timing metrics
        wandb_metrics.update({
            'time/train_transfer_to_device': epoch_timing_stats['train_transfer_to_device'],
            'time/train_forward_pass': epoch_timing_stats['train_forward_pass'],
            'time/train_backward_pass': epoch_timing_stats['train_backward_pass'],
            'time/train_optimizer_step': epoch_timing_stats['train_optimizer_step'],
            'time/train_misc_operations': epoch_timing_stats['train_misc_operations'],
            'time/train_unaccounted': train_unaccounted,
            'time/dataloader_overhead': epoch_timing_stats['dataloader_overhead'],
            'time/validation_transfer_to_device': epoch_timing_stats['validation_transfer_to_device'],
            'time/validation_forward_pass': epoch_timing_stats['validation_forward_pass'],
            'time/validation_misc_operations': epoch_timing_stats['validation_misc_operations'],
            'time/validation_unaccounted': val_unaccounted,
            'time/epoch_overhead': epoch_timing_stats['epoch_overhead'],
            'time/total_train_time': total_train_time,
            'time/total_validation_time': total_validation_time,
            'time/total_epoch_time': total_epoch_time
        })
        
        # Log to wandb
        wandb.log(wandb_metrics)
        
        # Detailed epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Q Loss: {train_q_loss:.4f}, Val Q Loss: {val_q_loss:.4f}")
        print(f"Train DT Loss: {train_dt_loss:.4f}, Val DT Loss: {val_dt_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Print timing information with percentages
        print("\nDetailed Timing Information (seconds):")
        print("Training:")
        print(f"  Data Transfer to Device: {epoch_timing_stats['train_transfer_to_device']:.2f} s ({100 * epoch_timing_stats['train_transfer_to_device'] / total_train_time:.1f}%)")
        print(f"  Forward Pass: {epoch_timing_stats['train_forward_pass']:.2f} s ({100 * epoch_timing_stats['train_forward_pass'] / total_train_time:.1f}%)")
        print(f"  Backward Pass: {epoch_timing_stats['train_backward_pass']:.2f} s ({100 * epoch_timing_stats['train_backward_pass'] / total_train_time:.1f}%)")
        print(f"  Optimizer Step: {epoch_timing_stats['train_optimizer_step']:.2f} s ({100 * epoch_timing_stats['train_optimizer_step'] / total_train_time:.1f}%)")
        print(f"  Misc Operations: {epoch_timing_stats['train_misc_operations']:.2f} s ({100 * epoch_timing_stats['train_misc_operations'] / total_train_time:.1f}%)")
        print(f"  Dataloader Overhead: {epoch_timing_stats['dataloader_overhead']:.2f} s ({100 * epoch_timing_stats['dataloader_overhead'] / total_train_time:.1f}%)")
        print(f"  Unaccounted Time: {train_unaccounted:.2f} s ({100 * train_unaccounted / total_train_time:.1f}%)")
        print(f"  Total Training Time: {total_train_time:.2f} s")
        
        print("Validation:")
        print(f"  Data Transfer to Device: {epoch_timing_stats['validation_transfer_to_device']:.2f} s ({100 * epoch_timing_stats['validation_transfer_to_device'] / total_validation_time:.1f}%)")
        print(f"  Forward Pass: {epoch_timing_stats['validation_forward_pass']:.2f} s ({100 * epoch_timing_stats['validation_forward_pass'] / total_validation_time:.1f}%)")
        print(f"  Misc Operations: {epoch_timing_stats['validation_misc_operations']:.2f} s ({100 * epoch_timing_stats['validation_misc_operations'] / total_validation_time:.1f}%)")
        print(f"  Unaccounted Time: {val_unaccounted:.2f} s ({100 * val_unaccounted / total_validation_time:.1f}%)")
        print(f"  Total Validation Time: {total_validation_time:.2f} s")
        
        print(f"Epoch Overhead: {epoch_timing_stats['epoch_overhead']:.2f} s")
        print(f"Total Epoch Time: {total_epoch_time:.2f} s")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            
            # Log best model as an artifact to wandb
            artifact = wandb.Artifact(f'best-model-{wandb.run.id}', type='model')
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
        else:
            no_improve_epochs += 1
            
        # Check for early stopping
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Print overall timing statistics
    print("\nOverall Training Timing Statistics (seconds):")
    total_time = sum(timing_stats.values())
    for key, value in sorted(timing_stats.items()):
        print(f"  {key}: {value:.2f} s ({100 * value / total_time:.1f}%)")
    print(f"  Average epoch time: {(sum(timing_stats.values()) / (epoch + 1)):.2f} s")
    
    # Create final timing chart for wandb
    wandb.log({
        "overall_time_breakdown": wandb.plot.bar(
            wandb.Table(
                columns=["Component", "Percentage"],
                data=[
                    ["Train Data Transfer", 100 * timing_stats['train_transfer_to_device'] / total_time],
                    ["Train Forward Pass", 100 * timing_stats['train_forward_pass'] / total_time],
                    ["Train Backward Pass", 100 * timing_stats['train_backward_pass'] / total_time],
                    ["Train Optimizer Step", 100 * timing_stats['train_optimizer_step'] / total_time],
                    ["Train Misc Operations", 100 * timing_stats['train_misc_operations'] / total_time],
                    ["Dataloader Overhead", 100 * timing_stats['dataloader_overhead'] / total_time],
                    ["Validation Time", 100 * (timing_stats['validation_transfer_to_device'] + 
                                              timing_stats['validation_forward_pass'] + 
                                              timing_stats['validation_misc_operations']) / total_time],
                    ["Epoch Overhead", 100 * timing_stats['epoch_overhead'] / total_time]
                ]
            ),
            "Component", "Percentage",
            title="Overall Time Breakdown (%)"
        )
    })

    # Return the model
    return model

def load_datasets_with_indices(cosmic_file, marley_file, pos_encoding_file, batch_size, num_workers, max_waveforms, indices_path):
    """Load datasets using pre-saved stratified split indices."""
    # Create cosmic dataloader
    cosmic_loader = create_dataloader(
        data_path=cosmic_file,
        pos_encoding_path=pos_encoding_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        max_waveforms=max_waveforms
    )
    
    # Create marley dataloader
    marley_loader = create_dataloader(
        data_path=marley_file,
        pos_encoding_path=pos_encoding_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        max_waveforms=max_waveforms
    )
    
    # Load saved indices
    split_indices = torch.load(indices_path)
    
    # Reconstruct subsets for each dataset type
    cosmic_train = torch.utils.data.Subset(cosmic_loader.dataset, split_indices['cosmic']['train_indices'])
    cosmic_val = torch.utils.data.Subset(cosmic_loader.dataset, split_indices['cosmic']['val_indices'])
    marley_train = torch.utils.data.Subset(marley_loader.dataset, split_indices['marley']['train_indices'])
    marley_val = torch.utils.data.Subset(marley_loader.dataset, split_indices['marley']['val_indices'])
    
    # Combine train and val sets
    train_dataset = torch.utils.data.ConcatDataset([cosmic_train, marley_train])
    val_dataset = torch.utils.data.ConcatDataset([cosmic_val, marley_val])
    
    return train_dataset, val_dataset

def main():
    args = parse_arguments()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # Initialize Model
    model = ModifiedTransformerModel(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(args.device)

    # Optimize CUDA operations if available
    if args.device == "cuda":
        print("Optimizing CUDA operations...")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Create dataloaders
    print("Initializing dataset...")
    start_time = time.time()
    
    # Create cosmic dataloader
    cosmic_loader = create_dataloader(
        data_path=args.cosmic_file,
        pos_encoding_path=args.pos_encoding_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        max_waveforms=args.max_waveforms
    )
    
    # Create marley dataloader
    marley_loader = create_dataloader(
        data_path=args.marley_file,
        pos_encoding_path=args.pos_encoding_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        max_waveforms=args.max_waveforms
    )
    
    print(f"Dataset initialization took {time.time() - start_time:.2f} seconds")
    print(f"Cosmic waveforms: {len(cosmic_loader.dataset)}")
    print(f"Marley waveforms: {len(marley_loader.dataset)}")

    # Split into training
    random_seed = 42  
    torch.manual_seed(random_seed)
    
    # Split cosmic dataset
    cosmic_size = len(cosmic_loader.dataset)
    cosmic_train_size = int(0.8 * cosmic_size)
    cosmic_val_size = cosmic_size - cosmic_train_size
    cosmic_train, cosmic_val = random_split(
        cosmic_loader.dataset,
        [cosmic_train_size, cosmic_val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Split marley dataset
    marley_size = len(marley_loader.dataset)
    marley_train_size = int(0.8 * marley_size)
    marley_val_size = marley_size - marley_train_size
    marley_train, marley_val = random_split(
        marley_loader.dataset,
        [marley_train_size, marley_val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Combine train and val sets
    train_dataset = torch.utils.data.ConcatDataset([cosmic_train, marley_train])
    val_dataset = torch.utils.data.ConcatDataset([cosmic_val, marley_val])
    
    # Save the indices 
    split_indices = {
        'cosmic': {
            'train_indices': cosmic_train.indices,
            'val_indices': cosmic_val.indices,
            'total_size': cosmic_size,
            'train_size': cosmic_train_size,
            'val_size': cosmic_val_size
        },
        'marley': {
            'train_indices': marley_train.indices,
            'val_indices': marley_val.indices,
            'total_size': marley_size,
            'train_size': marley_train_size,
            'val_size': marley_val_size
        },
        'random_seed': random_seed,
        'split_ratio': [0.8, 0.2]
    }
    
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    indices_path = os.path.join(args.checkpoint_dir, 'split_indices.pt')
    torch.save(split_indices, indices_path)
    print(f"Saved stratified split indices to {indices_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Cosmic - Train: {len(cosmic_train)}, Val: {len(cosmic_val)}")
    print(f"Marley - Train: {len(marley_train)}, Val: {len(marley_val)}")
    print(f"Total - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=True  
    )

    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )

    # Save Final Model
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")
    
    # Log final model and split indices as artifacts
    artifact = wandb.Artifact(f'final-model-{wandb.run.id}', type='model')
    artifact.add_file(final_model_path)
    artifact.add_file(indices_path)
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    main()