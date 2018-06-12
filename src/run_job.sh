#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # number of MPI processes
#SBATCH --cpus-per-task=24      # 24 cores on cedar nodes
#SBATCH --gres=gpu:4            # special request for GPUs
#SBATCH --account=rrg-hamarneh
#SBATCH --mem=0                 # give all memory you have in the node
#SBATCH --time=50:00:00         # time (DD-HH:MM)
#SBATCH -output=slurm.%N.%j.out      # STDOUT
#SBATCH --job-name=train_seg.py
#SBATCH --mail-user=asa224@sfu.ca
#SBATCH --mail-type=ALL

# load the modules
echo "Loading Python module.."
module load python/2.7.14

# change to appropriate environment
echo "Changing to brats environment.."
workon brats

# run the command
python train_seg.py --e 50 --b 10 --dm isensee --o isensee_main
