#!/bin/bash
#SBATCH --job-name=mae_classifier
#SBATCH --output=mae_classifier_50_%j.out
#SBATCH --error=mae_classifier_50_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=wongjiradlab

# Start timing the job
START_TIME=$(date +%s)

# Print some information about the job
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Container path
CONTAINER=/cluster/tufts/wongjiradlabnu/coherent/singularity-geant4-10.02.p03-ubuntu20.02.simg

# Load required modules
echo "Loading modules..."
module load singularity/3.6.1
module load python/3.8
module load cuda/11.7
echo "Modules loaded successfully."

# Define paths
WORKDIR=/cluster/tufts/wongjiradlabnu/vdasil01/ML_work
OUTPUT_DIR="${WORKDIR}/checkpoints_classifier_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4
export NCCL_DEBUG=INFO

# Print GPU information
echo "GPU information:"
nvidia-smi

# Print Python and PyTorch information
echo "Python and PyTorch information:"
singularity exec --nv $CONTAINER python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
"

# Change to working directory
cd $WORKDIR

# Verify required files exist
echo "Verifying required files..."
required_files=(
    "Classifier_Model.py" 
    "Transformer_Model_multi_residual.py"
    "Classifier_Dataset.py"
    "checkpoints_resumed_50_100_LR_5e5/model_epoch_100.pth"
    "cosmic_events_processed.h5"
    "marley_events_processed.h5" 
    "position_encodings.npz"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
        echo "ERROR: Required file $file not found. Exiting."
        exit 1
    fi
done

echo "Starting MAE Binary Classifier training with 4 GPUs..."
echo "Output directory: $OUTPUT_DIR"

# Run the training script within the Singularity container
#singularity exec --nv $CONTAINER python3 Classifier_Train_multi.py 
singularity exec --nv -B $WORKDIR $CONTAINER bash -c "cd $WORKDIR && python3 Classifier_Train_multi.py"
#singularity exec --nv -B $WORKDIR $CONTAINER bash -c "cd $WORKDIR && python3 Classifier_Train_wandb.py"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi

# Copy checkpoints to output directory if they exist
if [ -d "checkpoints_max_distributed" ]; then
    echo "Copying checkpoints to output directory..."
    cp -r checkpoints_max_distributed/* $OUTPUT_DIR/
    echo "Checkpoints copied to: $OUTPUT_DIR"
fi

# Print memory and resource usage summary
echo "Resource usage summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,AveRSS,AveVMSize,ReqMem,AllocCPUS,ReqGRES

# Calculate and display job execution time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job execution summary:"
echo "- Start time: $(date -d @$START_TIME)"
echo "- End time: $(date -d @$END_TIME)" 
echo "- Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "- Output directory: $OUTPUT_DIR"
echo "- Checkpoints location: checkpoints_max_distributed/"

echo "Job completed successfully at: $(date)"
