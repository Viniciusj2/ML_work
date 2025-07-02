#!/bin/bash
#SBATCH --job-name=mae_transformer
#SBATCH --output=mae_cluster_res_%j.out
#SBATCH --error=mae_cluster_res_%j.err
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

# Load Singularity module
echo "Loading Singularity module..."
module load singularity/3.5.3
echo "Singularity module loaded."

echo "Loading Python module..."
module load python/3.8
module load cuda/11.7
module spider cudnn
module spider pytorch
echo "Modules loaded."

# Define paths
WORKDIR=/cluster/tufts/wongjiradlabnu/vdasil01/ML_work
OUTPUT_DIR="${WORKDIR}/checkpoints_Test/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
DATA_DIR=${WORKDIR}

# Run the training script within the Singularity container
echo "Starting training..."
singularity exec --nv $CONTAINER python3 Transformer_Train_multi.py \
--cosmic_file ${DATA_DIR}/cosmic_events_processed.h5 \
--marley_file ${DATA_DIR}/marley_events_processed.h5 \
--pos_encoding_file ${DATA_DIR}/position_encodings.npz \
--output_dir $OUTPUT_DIR \
--batch_size 64 \
--epochs 50 \
--lr 5e-4 \
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
--warmup_epochs 3 \
--wandb \
--project "pmt-mae-transformer" \
--run_name "50-epoch-residual-cluster-run"

echo "Memory usage summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,AveRSS,AveVMSize
echo "Job finished at: $(date)"

# Calculate execution time and print it
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo "Total job execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Job finished at: $(date)"
