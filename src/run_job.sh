#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --gres=gpu:gpu:4     # Ask for 4 GPUs per node of the large-gpu node variety
#SBATCH --time=48:00:00
#SBATCH --job-name=train_seg.py
#SBATCH --account=rrg-hamarneh
#SBATCH --mem=128000M
#SBATCH --mail-user=asa224@sfu.ca
#SBATCH --mail-type=ALL
echo "Creating new folder in shared memory space at the node"
mkdir /dev/shm/asa224/
echo "Created new folder!"
echo "Copying the database file from /scratch to /dev/shm/asa224/"
cp -v /home/asa224/scratch/asa224/Datasets/BRATS2018/HDF5_Datasets/BRATS2018.h5 /dev/shm/asa224
echo "Copy complete!"
module load python/2.7.13
source ~/.bashrc
workon brats
pip freeze
echo "hello"
which python
python train_seg.py --epochs 40 --batch-size 2 --hdf5 /dev/shm/asa224/BRATS2018.h5
