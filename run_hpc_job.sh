#!/bin/bash
#SBATCH --job-name=phi35_seq_tune
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00

set -e

mkdir -p logs
mkdir -p artifacts
mkdir -p outputs

module load miniconda/24.4.0

PYTHON_BIN="/home/sfa631/.conda/envs/nlp_project/bin/python"

cd ~/hpc_json_project

echo "=== Environment Check ==="
hostname
nvidia-smi

echo "=== Python Check ==="
$PYTHON_BIN -c "import sys; print(sys.executable)"
$PYTHON_BIN -c "import yaml; print('yaml ok')"

echo "=== Stage 1: Alpaca Fine-Tuning ==="
$PYTHON_BIN scripts/train_stage1.py > logs/train_stage1.out 2>&1

echo "=== Checkpoint 1 Evaluation ==="
$PYTHON_BIN evaluation/run_checkpoint1_eval.py > logs/eval_checkpoint1.out 2>&1

echo "=== Stage 2: JSON Fine-Tuning ==="
$PYTHON_BIN scripts/train_stage2.py > logs/train_stage2.out 2>&1

echo "=== Checkpoint 2 Evaluation ==="
$PYTHON_BIN evaluation/run_checkpoint2_eval.py > logs/eval_checkpoint2.out 2>&1

echo "=== Base Model Evaluation (Checkpoint 0) ==="
$PYTHON_BIN evaluation/run_checkpoint0_eval.py > logs/eval_checkpoint0.out 2>&1

echo "=== Alpaca Automatic Metrics ==="
$PYTHON_BIN evaluation/score_alpaca_metrics.py > logs/alpaca_metrics.out 2>&1

echo "=== Job Complete ==="
