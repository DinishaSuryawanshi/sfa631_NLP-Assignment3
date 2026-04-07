# Assignment 3: Sequential Instruction Tuning of a Small LLM

**LLM & Agentic Systems — UTSA Graduate Course**
This report is published as a GitHub blog-style submission in accordance with assignment requirements.
---
## 1. Methodology

### Student Model Selection

This project uses **Phi-3.5-Mini-Instruct** as the student model. The model was selected due to its ability to run efficiently under QLoRA on UTSA HPC while still providing strong instruction-following performance. Its relatively small size allows for multi-stage fine-tuning within GPU memory constraints while maintaining meaningful evaluation signal across both general and structured tasks.

---

### Stage 1 Data: Alpaca Instruction Dataset
Stage 1 fine-tuning uses an Alpaca-style dataset consisting of general instruction-response pairs. The goal of this stage is to improve the model’s ability to follow natural language instructions across a wide variety of domains.

Each sample contains:
- Instruction
- Input (optional)
- Output

The dataset is converted into a chat-style format before training.

---

### Stage 2 Data: JSON Imitation Learning Dataset

Stage 2 data was generated using a teacher model (**Llama 3.3 70B Instruct**) via the UTSA API. The dataset covers five structured-output task types:

- Extraction  
- Schema adherence  
- Classification  
- JSON repair  
- Tool-use generation  

Each generated output was validated using `json.loads()`. Only syntactically valid JSON outputs were retained.

The final dataset consisted of **11 validated examples**.

---

### Why the Dataset Contains Only 11 Examples

The small dataset size was the result of **intentionally strict validation**, not generation failure.

Each teacher-generated output had to:
1. Parse successfully with `json.loads()`
2. Contain no markdown or extra text
3. Follow the intended structure exactly
4. Align with the task definition

Many generated responses were discarded due to:
- extra explanation text
- markdown code blocks
- malformed JSON
- schema inconsistencies
- incorrect key/value structures

This filtering prioritized **high-quality structured outputs over dataset size**.

However, this decision introduced a tradeoff:

> A small but clean dataset may not provide enough learning signal to influence model behavior.

This tradeoff directly impacted Stage 2 effectiveness.

---

### Training Configuration

Training was performed using **QLoRA with 4-bit quantization**.

Two-stage fine-tuning:
- **Stage 1:** Alpaca instruction tuning  
- **Stage 2:** JSON specialization  

Stage 2 continued from Stage 1 weights.

---

### UTSA HPC Setup

Training was executed on UTSA HPC using Slurm batch scheduling.

Each job:
- requested 1 GPU (V100)
- used a Conda environment
- was launched via `sbatch`

Artifacts collected:
- logs
- loss curves
- trainer state files
- GPU verification outputs

---

## 2. Experiments

### 2.1 Three-Checkpoint Comparison

| Model | ROUGE-L | BERTScore F1 | JSON Validity | Exact Match      
|------|--------|-------------|---------------|-------------|
| **C0 (Base)** | 0.2712      | 0.8781        | 9.1%        | 9.1%  
| **C1 (Alpaca)** | 0.4523    | 0.9178        | 72.7%       | 45.5% 
| **C2 (JSON)** | 0.4523      | 0.9178        | 72.7%       | 45.5% 

---

### 2.2 Alpaca Evaluation Results

| Comparison | A Wins | B Wins | Tie |
|------------|--------|--------|-----|
| **C0 vs C1** | 42% | 22% | 36% |
| **C1 vs C2** | 0% | 0% | 100% |

---

### 2.3 JSON Structured Output Evaluation

| Model | JSON Validity | Exact Match |
|------|--------------|------------|
| **C0** | 9.1% | 9.1% |
| **C1** | 72.7% | 45.5% |
| **C2** | 72.7% | 45.5% |

---

### 2.4 Forgetting Analysis

Comparison between C1 and C2:

- ROUGE-L: unchanged  
- BERTScore: unchanged  
- Judge evaluation: 100% ties  
- Outputs: identical across all prompts  

Conclusion:

> No catastrophic forgetting was observed.

---

### 2.5 Ablation Study

A constrained ablation was performed comparing C1 and C2.

#### Results

| Metric | C1 | C2 | Change |
|------|------|------|------|
| ROUGE-L | 0.4523 | 0.4523 | 0.0000 |
| BERTScore F1 | 0.9178 | 0.9178 | 0.0000 |
| JSON Validity | 72.7% | 72.7% | 0.0% |
| Exact Match | 45.5% | 45.5% | 0.0% |

Additional observations:
- Outputs identical across all Alpaca prompts
- Judge evaluation: 100% ties

#### Interpretation

Stage 2 fine-tuning produced a new checkpoint, but did not change model behavior.

The most likely explanation is that the dataset (11 examples) was too small to produce a meaningful gradient signal.

Because Stage 1 already significantly improved performance, the additional Stage 2 updates were insufficient to shift output behavior.

#### Key Insight

> Sequential fine-tuning does not guarantee improvement when the second-stage dataset is too small.

---

### Visual Summary

```mermaid
xychart-beta
    title "General Task Performance"
    x-axis ["C0", "C1", "C2"]
    y-axis "Score" 0 --> 1
    bar "ROUGE-L" [0.2712, 0.4523, 0.4523]
    bar "BERTScore F1" [0.8781, 0.9178, 0.9178]

Github Link: https://github.com/DinishaSuryawanshi/sfa631_NLP-Assignment3