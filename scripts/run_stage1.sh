#!/bin/bash

# Load conda
module load miniconda3

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp_project

# Go to project directory
cd ~/hpc_json_project

# Run Stage 1 training
python scripts/train_stage1.py
