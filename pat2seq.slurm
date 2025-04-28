#!/bin/bash
#SBATCH --job-name=pat2seq
#SBATCH --output=pat2seq.out
#SBATCH --error=pat2seq.err
#SBATCH --partition=ou_bcs_low
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8

source /home/gokulg/miniforge3/etc/profile.d/conda.sh
conda activate wgbs
python datasets/preprocessing/pat_to_seq.py
