#!/bin/bash
#SBATCH --job-name=mae_resume
#SBATCH --output=mae_resume_%j.out
#SBATCH --error=mae_resume_%j.err
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=wongjiradlab

# Start timing the job
START_TIME=$(date +%s)

# Print some information about the job
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"

CONTAINER=/cluster/tufts/wongjiradlabnu/coherent/singularity-geant4-10.02.p03-ubuntu20.02.simg

# Load modules
echo "Loading modules..."
module load singularity/3.5.3
module load python/3.8
module load cuda/11.7
echo "Modules loaded."

# Define paths
WORKDIR=/cluster/tufts/wongjiradlabnu/vdasil01/ML_work
DATA_DIR=${WORKDIR}

# CHECKPOINT CONFIGURATION - MODIFY THESE LINES
#CHECKPOINT_PATH="${WORKDIR}/checkpoints_Test/100_epochs/model_epoch_100.pth"
CHECKPOINT_PATH="${WORKDIR}/checkpoints_resumed_50_100_LR_5e5/model_epoch_100.pth"
OUTPUT_DIR="${WORKDIR}/checkpoints_resumed_100_120_LR_2e5"
mkdir -p $OUTPUT_DIR

# Optional overrides - uncomment and modify as needed
OVERRIDE_LR="--override_lr 2e-5"  # e.g., "--override_lr 5e-5"
OVERRIDE_MASK_RATIO="--override_mask_ratio 0.50"  # e.g., "--override_mask_ratio 0.%%"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

# Run the training script
echo "Starting resumed training..."
singularity exec --nv $CONTAINER python3 Transformer_Train_multi_reload.py \
--resume $CHECKPOINT_PATH \
$OVERRIDE_LR \
$OVERRIDE_MASK_RATIO \
--cosmic_file ${DATA_DIR}/cosmic_events_processed.h5 \
--marley_file ${DATA_DIR}/marley_events_processed.h5 \
--pos_encoding_file ${DATA_DIR}/position_encodings.npz \
--output_dir $OUTPUT_DIR \
--batch_size 64 \
--epochs 120 \
--lr 1e-4 \
--weight_decay 1e-5 \
--mask_ratio 0.5 \
--encoder_dim 256 \
--decoder_dim 128 \
--encoder_heads 16 \
--decoder_heads 8 \
--encoder_layers 12 \
--decoder_layers 6 \
--encoder_mlp_ratio 16 \
--decoder_mlp_ratio 8 \
--dropout 0.1 \
--num_workers 2 \
--warmup_epochs 0 \
--wandb \
--project "pmt-mae-transformer" \
--run_name "resumed-cluster-debug-$(date +%Y%m%d_%H%M%S)"

echo "Memory usage summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,AveRSS,AveVMSize
echo "Job finished at: $(date)"

# Calculate execution time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo "Total job execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
