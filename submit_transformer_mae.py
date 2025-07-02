#!/bin/bash
#SBATCH --job-name=mae_transformer
#SBATCH --output=mae_cluster_%j.out
#SBATCH --error=mae_cluster_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
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

# Load only the Python module - we'll use packages from inside the container
echo "Loading Python module..."
module load python/3.8
module load cuda/11.7
# Try to find the correct cudnn version
module spider cudnn
# Try to find the correct PyTorch version  
module spider pytorch
echo "Modules loaded."

# Define paths
WORKDIR=/cluster/tufts/wongjiradlabnu/vdasil01/ML_work
OUTPUT_DIR="${WORKDIR}/checkpoints_classifier/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
DATA_DIR=${WORKDIR}

# Run the training script within the Singularity container
echo "Starting training..."
singularity exec --nv $CONTAINER python3 Classifier_Train.py \
--project "pmt-mae-classifier" \
--run_name "Cluster_test_run"

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
