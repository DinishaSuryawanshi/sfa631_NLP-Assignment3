# Assignment 3: Sequential Instruction Tuning of a Small LLM

This repository contains a two-stage post-training pipeline for a small instruction-tuned language model on UTSA HPC.

## Project Summary

The pipeline implements:

1. **Phase 1**: Data construction
   - Alpaca-style instruction dataset preparation
   - Teacher-generated JSON imitation-learning dataset construction

2. **Phase 2**: Stage 1 fine-tuning on Alpaca data
   - QLoRA fine-tuning on UTSA HPC
   - Save Checkpoint 1

3. **Phase 3**: Stage 2 fine-tuning on teacher-generated JSON data
   - Continue training from Checkpoint 1
   - Save Checkpoint 2

4. **Phase 4**: Evaluation and forgetting analysis
   - Checkpoint 0, Checkpoint 1, Checkpoint 2
   - Alpaca automatic metrics
   - JSON validity / exact match
   - Judge-based pairwise comparison

## Student Model

- `microsoft/Phi-3.5-mini-instruct`

## Teacher / Judge Model

- `llama-3.3-70b-instruct-awq`
- Accessed through the UTSA API endpoint

## Repository Structure

```text
config/
data/
evaluation/
logs/
outputs/
prompts/
scripts/
README.md
requirements.txt
run_hpc_job.sh

## Run Batch Job
sbatch run_hpc_job.sh