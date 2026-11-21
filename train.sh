#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=alexnet
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd /scratch/$USER/reproducing_alexnet_on_CIFAR-10
export UV_CACHE_DIR=/scratch/$USER/.uv-cache
uv run python scripts/run_ablations.py