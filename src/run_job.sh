#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --gres=gpu:lgpu:4     # Ask for 4 GPUs per node of the large-gpu node variety
#SBATCH --time=2-12:00
#SBATCH --job-name=train_seg.py
#SBATCH --account=rrg-hamarneh
#SBATCH --mem=256000M
#SBATCH --mail-user=asa224@sfu.ca
#SBATCH --mail-type=ALL
module load python/2.7.13
source ~/.bashrc
workon brats
pip freeze
echo "hello"
which python
python train_seg.py --epochs 40 --batch-size 2

