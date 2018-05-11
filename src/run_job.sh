#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --gres=gpu:lgpu:4     # Ask for 4 GPUs per node of the large-gpu node variety
#SBATCH --time=1-00:00
#SBATCH --account=rrg-hamarneh
#SBATCH --mem=32000M

module load python/2.7.13
workon brats
python train_seg.py --epochs 20

