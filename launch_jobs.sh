#!/bin/bash
# launch_jobs.sh

data_dir='/orcd/data/omarabu/001/gokul/DistributionEmbeddings/data/geo_downloads'

for f in "$data_dir"/*.hg38.pat.gz; do
    outfile="${f/.hg38.pat.gz/_seqs.csv}"

    sbatch --job-name=pat2seq_$(basename "$f" .hg38.pat.gz) \
           --output=slurm_logs/%x_%j.out \
           --error=slurm_logs/%x_%j.err \
           --partition=ou_bcs_low \
           --mem=32G \
           --time=1:00:00 \
           --cpus-per-task=1 \
           --wrap="source /home/gokulg/miniforge3/etc/profile.d/conda.sh && conda activate wgbs && python datasets/preprocessing/process_one.py \"$f\""
done
