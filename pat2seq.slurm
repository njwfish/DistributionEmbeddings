#!/bin/bash
#SBATCH --job-name=pat2seq
#SBATCH --output=pat2seq.out
#SBATCH --error=pat2seq.err
#SBATCH --partition=ou_bcs_low
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1

source /home/gokulg/miniforge3/etc/profile.d/conda.sh
conda activate wgbs
python datasets/preprocessing/pat_to_seq.py
