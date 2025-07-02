import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import wandb
import os
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import time
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from Transformer_Dataset_multi import MAEPMTDataset 
from Transformer_Dataset_multi import create_mae_dataloader, split_and_combine_dataloaders
from Transformer_Model_multi_residual import MAETransformerModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging 

def parse_args():
    parser = argparse.ArgumentParser(description="Train MAE Transformer for PMT data")
    
    # Data paths
    parser.add_argument('--cosmic_file', type=str, required=False, default='preprocessed_data/cosmic_events_processed.h5', help='Path to preprocessed cosmic data file')
    parser.add_argument('--marley_file', type=str, required=False, default='preprocessed_data/marley_events_processed.h5', help='Path to preprocessed marley data file')
    parser.add_argument('--pos_encoding_file', type=str, required=False, default='preprocessed_data/position_encodings.npz', help='Path to position encodings file')
    parser.add_argument("--output_dir", type=str, default="./checkpoints_residual_50", help="Directory to save checkpoints")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--mask_ratio", type=float, default=0.50, help="Ratio of tokens to mask")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=84, help="Random seed")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with subset of data")
    parser.add_argument("--max_waveforms", type=int, default=5, 
                        help="Maximum number of waveforms to use when debug mode is enabled")
    
    # Model parameters
    parser.add_argument("--encoder_dim", type=int, default=256, help="Encoder embedding dimension")
    parser.add_argument("--decoder_dim", type=int, default=128, help="Decoder embedding dimension")
    parser.add_argument("--encoder_heads", type=int, default=16, help="Number of encoder attention heads")
    parser.add_argument("--decoder_heads", type=int, default=8, help="Number of decoder attention heads")
    parser.add_argument("--encoder_layers", type=int, default=12, help="Number of encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--encoder_mlp_ratio", type=int, default=16, help="Encoder MLP ratio")
    parser.add_argument("--decoder_mlp_ratio", type=int, default=8, help="Decoder MLP ratio")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # New feature separation parameters
    #parser.add_argument("--spatial_dim", type=int, default=17, help="Dimension of spatial features (8 for x + 8 for y + 1 for z)")
    #parser.add_argument("--non_spatial_dim", type=int, default=2, help="Dimension of non-spatial features (1 for charge + 1 for delta time)")
    
    # W&B parameters
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--project", type=str, default="pmt-mae-transformer", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--warmup_epochs", type=int, default=4, help="Number of warmup epochs for LR scheduler")
    
    return parser.parse_args()

# First time using this type of lr shceduler seems to be relevany for MAE taks
def get_lr_scheduler(optimizer, args, steps_per_epoch):
    """Create a learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        # Warmup phase
        warmup_steps = args.warmup_epochs * steps_per_epoch
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(
            max(1, args.epochs * steps_per_epoch - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, distributed=False):
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, distributed=False):
    """Run one training epoch"""
    model.train()
    
    # SET EPOCH FOR DISTRIBUTED SAMPLER
    if distributed and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

    epoch_start = time.time()
    running_loss = 0.0
    running_q_loss = 0.0
    running_dt_loss = 0.0
    valid_positions_total = 0
    
    # For plotting learning rate
    lr_history = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
    for i, batch in enumerate(pbar):
        # Move data to device
        total_pmts = batch['attention_mask'].sum(dim=1)  # [B]
        masked_pmts = batch['pred_mask'].sum(dim=1)
        visible_pmts = (batch['attention_mask'] & ~batch['pred_mask']).sum(dim=1)
        
        #Debugging ingo 
        # print(f"\nBatch verification:")
        # print(f"Ground truth PMTs: {total_pmts.tolist()}")
        # print(f"Masked PMTs: {masked_pmts.tolist()}")
        # print(f"Visible PMTs: {visible_pmts.tolist()}")
        # print(f"Sum check: {(masked_pmts + visible_pmts == total_pmts).all()}")

        event_tensor = batch["event_tensor"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pred_mask = batch["pred_mask"].to(device)
        original_values = batch["original_values"].to(device)
        
        # Forward pass
        outputs = model(event_tensor, attention_mask, pred_mask)
        
        # Compute loss
        #loss, loss_dict = model.compute_loss(
        #loss, loss_dict = model.module.compute_loss(
        loss, loss_dict = (model.module if distributed else model).compute_loss(
            outputs["output"],
            original_values,
            pred_mask,
            attention_mask
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
  
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
        
        # Update running metrics
        running_loss += loss_dict["total_loss"].item()
        running_q_loss += loss_dict["q_loss"].item()
        running_dt_loss += loss_dict["dt_loss"].item()
        valid_positions_total += loss_dict["valid_positions"]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{running_loss/(i+1):.4f}",
            'lr': f"{current_lr:.6f}"
        })
    
    # Calculate epoch metrics
    avg_loss = running_loss / len(dataloader)
    avg_q_loss = running_q_loss / len(dataloader)
    avg_dt_loss = running_dt_loss / len(dataloader)
    avg_lr = sum(lr_history) / len(lr_history)
    
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch} completed in {epoch_time:.2f}s | "
          f"Train Loss: {avg_loss:.4f} | "
          f"Q Loss: {avg_q_loss:.4f} | "
          f"DT Loss: {avg_dt_loss:.4f} | "
          f"LR: {avg_lr:.6f}")
    
    return {
        "train_loss": avg_loss,
        "train_q_loss": avg_q_loss,
        "train_dt_loss": avg_dt_loss,
        "train_valid_positions": valid_positions_total,
        "epoch_time": epoch_time,
        "learning_rate": avg_lr
    }

def validate(model, dataloader, device, epoch):
    """Run validation"""
    model.eval()
    val_start = time.time()
    running_loss = 0.0
    running_q_loss = 0.0
    running_dt_loss = 0.0
    valid_positions_total = 0
    
    # For visualization 
    sample_inputs = []
    sample_masks = []
    sample_outputs = []
    sample_targets = []
    sample_attention_masks = [] 
    max_samples = 50  

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
                        # Verification
            total = batch['attention_mask'].sum().item()
            masked = batch['pred_mask'].sum().item()
            visible = (batch['attention_mask'] & ~batch['pred_mask']).sum().item()
            
            #Debugging 
            # print(f"\nValidation Check:")
            # print(f"Total PMTs: {total} | "
            #         f"Masked: {masked} ({masked/total:.1%}) | "
            #         f"Visible: {visible} | "
            #         f"Sum: {masked + visible}/{total}")

            # if masked + visible != total:
            #     print(" WARNING: Masking inconsistency detected!")
                
            # Move data to device
            event_tensor = batch["event_tensor"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pred_mask = batch["pred_mask"].to(device)
            original_values = batch["original_values"].to(device)
            
            # Forward pass
            outputs = model(event_tensor, attention_mask, pred_mask)
            
            # Compute loss
            #_, loss_dict = model.compute_loss(
            #_, loss_dict = model.module.compute_loss(
            _, loss_dict = (model.module if hasattr(model, 'module') else model).compute_loss(
                outputs["output"],
                original_values,
                pred_mask,
                attention_mask
            )
            
            # Update running metrics
            running_loss += loss_dict["total_loss"].item()
            running_q_loss += loss_dict["q_loss"].item()
            running_dt_loss += loss_dict["dt_loss"].item()
            valid_positions_total += loss_dict["valid_positions"]
            
            # Store samples for visualization 
            if len(sample_inputs) < max_samples:
                # Get number of samples we need
                num_to_take = min(max_samples - len(sample_inputs), event_tensor.size(0))
                
                # Add samples to our lists
                sample_inputs.extend([event_tensor[i].cpu().numpy() for i in range(num_to_take)])
                sample_masks.extend([pred_mask[i].cpu().numpy() for i in range(num_to_take)])
                sample_outputs.extend([outputs["output"][i].cpu().numpy() for i in range(num_to_take)])
                sample_targets.extend([original_values[i].cpu().numpy() for i in range(num_to_take)])
                sample_attention_masks.extend([batch['attention_mask'][i].cpu().numpy() for i in range(num_to_take)])  # NEW LINE
    
    # Calculate validation metrics
    avg_loss = running_loss / len(dataloader)
    avg_q_loss = running_q_loss / len(dataloader)
    avg_dt_loss = running_dt_loss / len(dataloader)
    
    val_time = time.time() - val_start
    
    print(f"Validation | Epoch {epoch} | "
          f"Loss: {avg_loss:.4f} | "
          f"Q Loss: {avg_q_loss:.4f} | "
          f"DT Loss: {avg_dt_loss:.4f} | "
          f"Time: {val_time:.2f}s")
    
    return {
        "val_loss": avg_loss,
        "val_q_loss": avg_q_loss,
        "val_dt_loss": avg_dt_loss,
        "val_valid_positions": valid_positions_total,
        "val_time": val_time,
        "samples": {
            "inputs": sample_inputs,
            "masks": sample_masks,
            "attention_masks": sample_attention_masks, 
            "outputs": sample_outputs,
            "targets": sample_targets
        }
    }

def log_wandb_metrics(train_metrics, val_metrics, epoch, args):
    """Log simplified metrics to wandb"""
    if not args.wandb:
        return
        
    # Log Metrics on Wandb
    metrics = {
        # Training losses
        "losses/train_q_loss": train_metrics["train_q_loss"],
        "losses/train_dt_loss": train_metrics["train_dt_loss"],
        "losses/train_total_loss": train_metrics["train_loss"],
        
        # Validation losses
        "losses/val_q_loss": val_metrics["val_q_loss"],
        "losses/val_dt_loss": val_metrics["val_dt_loss"], 
        "losses/val_total_loss": val_metrics["val_loss"],
        
        # Training metadata
        "training/epoch": epoch,
        "training/learning_rate": train_metrics["learning_rate"],
        "training/epoch_time": train_metrics["epoch_time"],
        
        # Model parameters
        "model/mask_ratio": args.mask_ratio,
    }
    
    # Make sure metrics are all scalar values (not tensors)
    for key, value in metrics.items():
        if torch.is_tensor(value):
            metrics[key] = value.item()
    
    # Explicitly log each metric
    wandb.log(metrics, step=epoch)

def log_wandb_reconstruction_examples(samples, epoch, args):
    """Log reconstruction examples with absolute error for masked regions only"""
    if not args.wandb:
        return
        
    try:
        import matplotlib.pyplot as plt
        
        for i in range(min(len(samples["inputs"]), 10)):  
            fig = plt.figure(figsize=(12, 8))
            
            # Get data
            target = samples["targets"][i][:, 0]  
            pred = samples["outputs"][i][:, 0]
            attention_mask = samples["attention_masks"][i]  # Valid PMTs
            pred_mask = samples["masks"][i]  # The prediction mask
            
            # Only consider valid PMTs that were masked
            valid_masked = attention_mask & pred_mask
            valid_masked_indices = np.where(valid_masked)[0]
            
            # Calculate absolute error for valid masked regions only
            if len(valid_masked_indices) > 0:
                abs_error = np.abs(pred[valid_masked_indices] - target[valid_masked_indices])
            
            # Create a subplot layout
            plt.subplot(2, 1, 1)
            
            # Plot all valid PMTs (ground truth)
            valid_indices = np.where(attention_mask)[0]
            plt.plot(valid_indices, target[valid_indices], 'ko', label='All Valid PMTs')
            
            # Plot visible PMTs (input to model)
            visible_indices = np.where(attention_mask & ~pred_mask)[0]
            plt.plot(visible_indices, target[visible_indices], 'bo', label='Visible Input')
            
            # Plot reconstructed PMTs (masked ones)
            if len(valid_masked_indices) > 0:
                plt.plot(valid_masked_indices, pred[valid_masked_indices], 'rx', label='Reconstructed')
            
            plt.title(f"Reconstruction Example (Epoch {epoch})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot the absolute error for masked regions only
            plt.subplot(2, 1, 2)
            
            if len(valid_masked_indices) > 0:
                plt.bar(valid_masked_indices, abs_error, color='r', alpha=0.7)
                plt.title(f"Absolute Error in Masked Regions (Mean: {np.mean(abs_error):.4f})")
                plt.xlabel("PMT Index")
                plt.ylabel("Absolute Error")
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, "No valid masked PMTs in this example", 
                         horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            
            # Log the figure to wandb - explicitly create the wandb.Image object
            wandb_img = wandb.Image(fig)
            wandb.log({f"reconstruction/example_{i+1}": wandb_img}, step=epoch)
            
            plt.close(fig)
            
        if samples["targets"][0].shape[1] > 1:
            for i in range(min(len(samples["inputs"]), 3)):
                fig = plt.figure(figsize=(12, 8))
                
                # Get dt data (second column)
                dt_target = samples["targets"][i][:, 1]
                dt_pred = samples["outputs"][i][:, 1]
                attention_mask = samples["attention_masks"][i]
                pred_mask = samples["masks"][i]
                
                # Only consider valid PMTs that were masked
                valid_masked = attention_mask & pred_mask
                valid_masked_indices = np.where(valid_masked)[0]
                
                if len(valid_masked_indices) > 0:
                    dt_abs_error = np.abs(dt_pred[valid_masked_indices] - dt_target[valid_masked_indices])
                
                # Plot dt data
                plt.subplot(2, 1, 1)

                valid_indices = np.where(attention_mask)[0]
                plt.plot(valid_indices, dt_target[valid_indices], 'ko', label='All Valid PMTs')
                
                # Plot visible PMTs (input to model)
                visible_indices = np.where(attention_mask & ~pred_mask)[0]
                plt.plot(visible_indices, dt_target[visible_indices], 'go', label='Visible Input')
                
                # Plot reconstructed PMTs (masked ones)
                if len(valid_masked_indices) > 0:
                    plt.plot(valid_masked_indices, dt_pred[valid_masked_indices], 'mx', label='Reconstructed')
                
                plt.title(f"DT Reconstruction (Epoch {epoch})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot the absolute dt error for masked regions
                plt.subplot(2, 1, 2)
                
                if len(valid_masked_indices) > 0:
                    plt.bar(valid_masked_indices, dt_abs_error, color='m', alpha=0.7)
                    plt.title(f"DT Absolute Error in Masked Regions (Mean: {np.mean(dt_abs_error):.4f})")
                    plt.xlabel("PMT Index")
                    plt.ylabel("Absolute Error")
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, "No valid masked PMTs in this example", 
                             horizontalalignment='center', verticalalignment='center')
                
                plt.tight_layout()
                
                # Log the dt figure to wandb - explicitly create the wandb.Image object
                wandb_img = wandb.Image(fig)
                wandb.log({f"reconstruction/dt_example_{i+1}": wandb_img}, step=epoch)
                
                plt.close(fig)
                
    except Exception as e:
        print(f"Error in reconstruction logging: {e}")
        import traceback
        traceback.print_exc()
def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def setup_for_distributed(is_master):
    """Disable printing when not in master process"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def distributed_main(rank, world_size, args):
    """Main function for distributed training"""
    # Setup distributed training
    setup_distributed(rank, world_size)
    setup_for_distributed(rank == 0)  # Only rank 0 prints
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    
    # Only initialize wandb on rank 0
    if args.wandb and rank == 0:
        try:
            import wandb
            run_name = args.run_name if args.run_name else f"mae-pmt-e{args.encoder_dim}-h{args.encoder_heads}-l{args.encoder_layers}-mr{args.mask_ratio}"
            wandb.init(
                project=args.project,
                name=run_name,
                config=vars(args),
                resume=False
            )
            print(f"W&B initialized successfully with run name: {run_name}")
        except Exception as e:
            print(f"Error initializing W&B: {e}")
            args.wandb = False
    
    # Set max_waveforms for debug mode
    max_waveforms = args.max_waveforms if args.debug else None
    
    # Create data loaders with distributed=True
    train_loader, val_loader = split_and_combine_dataloaders(
        cosmic_data_path=args.cosmic_file,
        marley_data_path=args.marley_file, 
        pos_encoding_path=args.pos_encoding_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        seed=args.seed,
        mask_ratio=args.mask_ratio,
        max_waveforms=max_waveforms,
        distributed=True  # ADD THIS LINE
    )
    
    # Create model
    dataset = train_loader.dataset.datasets[0].dataset
    spatial_dim = dataset.spatial_dim
    non_spatial_dim = dataset.non_spatial_dim
    
    model = MAETransformerModel(
        input_dim=spatial_dim + non_spatial_dim,
        encoder_embedding_dim=args.encoder_dim,
        decoder_embedding_dim=args.decoder_dim,
        encoder_num_heads=args.encoder_heads,
        decoder_num_heads=args.decoder_heads,
        encoder_num_layers=args.encoder_layers,
        decoder_num_layers=args.decoder_layers,
        encoder_mlp_ratio=args.encoder_mlp_ratio,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        dropout=args.dropout
    ).to(device)
    
    # WRAP MODEL WITH DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = get_lr_scheduler(optimizer, args, len(train_loader))
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train one epoch
        #train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, distributed=True)
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, args, distributed=True)
   
        # Validate
        val_metrics = validate(model, val_loader, device, epoch)
        
        # Only log on rank 0
        if args.wandb and rank == 0:
            try:
                log_wandb_metrics(train_metrics, val_metrics, epoch, args)
                log_wandb_reconstruction_examples(val_metrics["samples"], epoch, args)
            except Exception as e:
                print(f"Error logging to W&B: {e}")
        
        # Save checkpoint only on rank 0
        if rank == 0:
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # Use .module for DDP
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")
    
    # Clean up
    cleanup_distributed()
    
    if args.wandb and rank == 0:
        wandb.finish()

def main():
    # Parse command line arguments
    global args
    args = parse_args()
    world_size = torch.cuda.device_count()
    
    # Create output directory first (before any training starts)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if world_size > 1:
        print(f"Found {world_size} GPUs. Starting distributed training...")
        mp.spawn(distributed_main, args=(world_size, args), nprocs=world_size, join=True)
        return  # EXIT HERE - don't continue with single GPU code
    
    # Single GPU training code (only runs if world_size <= 1)
    print("Single GPU training...")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "training.log"))
        ]
    )
    logger = logging.getLogger(__name__)
    
    if torch.cuda.is_available():
        # This was recommended by Claude for faster training 
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize W&B - make sure to update the init call
    if args.wandb:
        try:
            import wandb
            
            run_name = args.run_name if args.run_name else f"mae-pmt-e{args.encoder_dim}-h{args.encoder_heads}-l{args.encoder_layers}-mr{args.mask_ratio}"
            wandb.init(
                project=args.project,
                name=run_name,
                config=vars(args),
                resume=False  # Make sure we're creating a new run
            )
            logger.info(f"W&B initialized successfully with run name: {run_name}")
            
            # Log the mask ratio explicitly in config
            logger.info(f"Using mask ratio: {args.mask_ratio}")
            
            # Log code to W&B
            wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            args.wandb = False  # Disable wandb if initialization fails
            logger.warning("Continuing without W&B logging")
    
    # Set max_waveforms for debug mode
    # In debug mode, we load at most args.max_waveforms from each dataset
    max_waveforms = args.max_waveforms if args.debug else None
    
    # Make this debug setting more explicit
    if args.debug:
        logger.info(f"DEBUG MODE ENABLED: Using max {max_waveforms} waveforms per dataset")
    else: 
        logger.info(f"FULL DATASET MODE: No maximum limit on waveforms (max_waveforms={max_waveforms})")
    
    # Create data loaders
    train_loader, val_loader = split_and_combine_dataloaders(
        cosmic_data_path=args.cosmic_file,
        marley_data_path=args.marley_file, 
        pos_encoding_path=args.pos_encoding_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        seed=args.seed,
        mask_ratio=args.mask_ratio,
        max_waveforms=max_waveforms  
    )
    
    # Get spatial and non-spatial dimensions from the dataset
    # We could use args.spatial_dim and args.non_spatial_dim, but extracting from the dataset
    # ensures consistency
    dataset = train_loader.dataset.datasets[0].dataset  # Access the underlying MAEPMTDataset
    spatial_dim = dataset.spatial_dim
    non_spatial_dim = dataset.non_spatial_dim
    
    logger.info(f"Dataset feature dimensions - Spatial: {spatial_dim}, Non-spatial: {non_spatial_dim}")
    
    # Create model with separated spatial and non-spatial features
    model = MAETransformerModel(
        input_dim=spatial_dim + non_spatial_dim,  # Total input dimension
      #  spatial_dim=spatial_dim,
      #  non_spatial_dim=non_spatial_dim,
        encoder_embedding_dim=args.encoder_dim,
        decoder_embedding_dim=args.decoder_dim,
        encoder_num_heads=args.encoder_heads,
        decoder_num_heads=args.decoder_heads,
        encoder_num_layers=args.encoder_layers,
        decoder_num_layers=args.decoder_layers,
        encoder_mlp_ratio=args.encoder_mlp_ratio,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = get_lr_scheduler(optimizer, args, len(train_loader))
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, args)
        # Validate
        val_metrics = validate(model, val_loader, device, epoch)
        
        # Log simplified metrics to wandb - add try-except block to catch potential errors
        if args.wandb:
            try:
                log_wandb_metrics(train_metrics, val_metrics, epoch, args)
                log_wandb_reconstruction_examples(val_metrics["samples"], epoch, args)
                logger.info("Successfully logged metrics and examples to W&B")
            except Exception as e:
                logger.error(f"Error logging to W&B: {e}")
                import traceback
                traceback.print_exc()
        
        metrics = {
            **train_metrics,
            **{k: v for k, v in val_metrics.items() if k != "samples"},
            "epoch": epoch
        }
        
        # Save checkpoint if validation loss improved
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
        
        if epoch % 10 == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics["val_loss"],
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
    
        if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")

    # Training finished
    total_time = time.time() - start_time
    logger.info(f"\nTraining finished in {total_time/60:.2f} minutes")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_metrics["val_loss"],
        'args': vars(args)
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Close wandb
    if args.wandb:
        try:
            wandb.finish()
            logger.info("W&B logging finished")
        except Exception as e:
            logger.error(f"Error finishing W&B run: {e}")   
      
if __name__ == "__main__":
    main()
