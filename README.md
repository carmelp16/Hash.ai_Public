# LIIH: Layered Integrity Invariant Hash Framework

A Python pipeline for detecting whether an LLM has been tampered with (fine-tuned, backdoored, quantized, or instruction-tuned) relative to a trusted baseline, using the **LIIH (Layered Integrity Invariant Hash) Framework**.

## Overview

LIIH frames tamper detection as a **pair classification problem**: given a trusted baseline model $V_1$ and a candidate model $V_2$, the pipeline extracts a 100-dimensional comparison feature vector and classifies the pair as *legitimate* (same model) or *tampered* (modified).

Detection targets:
- **Fine-tuning / RLHF**: Unauthorized weight updates or alignment changes
- **Backdoor injection**: BadNet-style poisoned models
- **Quantization**: Precision modifications (FP32 → INT8)
- **Instruction-tuning**: Base → instruction-tuned variant swaps
- **Cross-family substitution**: Model replaced with a different architecture

## Architecture

Four orthogonal detection components, each contributing comparison features to the final vector:

### Component I: Intrinsic Gradient Signatures (Jacobian) — 55 features
- Zeroth-order Jacobian estimation via **ZeroPrint** (arXiv:2510.06605)
- Semantic perturbations using DistilBERT as perturber
- Per-probe cosine similarities + delta mean vector + scalar statistics
- Detects: Model substitution, deep fine-tuning, weight-level changes

### Component B/S: Behavioral & Semantic Signatures — 28 features
- Free-text response embeddings via `all-MiniLM-L6-v2`
- Linguistic complexity analysis (sentence length, vocabulary richness)
- Per-probe cosine similarities + cross-mean similarity + complexity deltas
- Detects: RLHF updates, alignment drift, behavioral shifts

### Component T: Temporal & Operational Signatures — 6 features
- **TTFT** (Time to First Token) and **OTPS** (Output Tokens Per Second)
- Hardware-agnostic ratios and relative differences
- Detects: Quantization, hardware substitution, infrastructure changes

### Component L: LLMmap Traces — 11 features
- Bilateral trace embeddings via **LLMmap** (arXiv:2407.15847)
- Query ∥ response embeddings using `multilingual-E5-large-instruct`
- Per-probe cosine similarities + scalar statistics
- Detects: Identity-level model swaps, architecture changes

## Directory Structure

```
src/
├── __init__.py
├── config.py                    # Model pairs, datasets, LIIH component configs
├── pipeline.py                  # Main orchestration script
├── data/
│   ├── model_loader.py          # HuggingFace model loader (incl. int8 quantization)
│   └── dataset_loader.py        # MMLU-Pro & TruthfulQA probe datasets
├── features/
│   ├── jacobian_extractor.py    # Component I: ZeroPrint Jacobian fingerprinting
│   ├── semantic_extractor.py    # Component B/S: Semantic drift detection
│   ├── temporal_extractor.py    # Component T: TTFT & OTPS profiling
│   ├── llmmap_extractor.py      # Component L: LLMmap bilateral traces
│   └── liih_builder.py          # Composite LIIH comparison vector builder
├── classifier/
│   ├── trainer.py               # Multi-classifier trainer (RF, XGBoost, SVM, LR)
│   └── evaluator.py             # Evaluation, confusion matrices, feature importance
└── utils/
    ├── perturber.py             # Semantic perturbation for ZeroPrint
    └── helpers.py               # Utility functions
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended; required for int8 quantization pairs)
- 16GB+ RAM

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `transformers` — HuggingFace Transformers
- `torch` — PyTorch
- `bitsandbytes` — INT8/INT4 quantization via CUDA kernels
- `sentence-transformers` — Sentence embeddings
- `scikit-learn` — ML algorithms (RF, SVM, LR)
- `xgboost` — XGBoost classifier
- `datasets` — HuggingFace Datasets

## Usage

### Quick Start

```bash
python src/pipeline.py
```

This will:
1. Load all model pairs defined in `config.py` (legitimate + tampered)
2. Inject in-pipeline backdoor pairs (BadNet-style)
3. Extract per-model LIIH signatures (Jacobian, Semantic, Temporal, LLMmap)
4. Build pairwise 100-feature comparison vectors
5. Train four classifiers (Random Forest, XGBoost, SVM, Logistic Regression)
6. Run component-level ablation study
7. Generate evaluation reports and figures in `results/`

### Configuration

Edit `src/config.py` to customize model pairs and LIIH component settings:

```python
JACOBIAN_CONFIG = {
    "k_top_tokens": 32,       # Top-K tokens for Jacobian approximation
    "num_probes": 20,          # Number of perturbation probes per model
    "perturber_model": "..."   # DistilBERT perturber model name
}

SEMANTIC_CONFIG = {
    "num_behavioral_probes": 20,   # Number of semantic probes per model
    "max_new_tokens": 64,
    "embedding_model": "all-MiniLM-L6-v2"
}

TEMPORAL_CONFIG = {
    "num_timing_probes": 10,   # Number of timing measurements per model
    "fixed_input_length": 32,
    "fixed_output_length": 32,
    "warmup_runs": 2
}
```

## Model Pairs

The pipeline evaluates **38 model pairs** across five modification types, spanning 82M to 7B parameters and eight model families (GPT-2, GPT-Neo, OPT, Pythia, BLOOM, SmolLM2, Qwen2.5, OLMo):

| Modification Type       | # Pairs | Label |
|------------------------|---------|-------|
| Identical / legitimate | 17      | 0     |
| Instruction-tuning     | 9       | 1     |
| Backdoor injection     | 7       | 1     |
| Quantization (INT8)    | 3       | 1     |
| Fine-tuning            | 2       | 1     |

**Train/test split**: 28 train / 10 test (stratified by modification type × label).

## Results

| Classifier        | CV Accuracy         | Test Accuracy |
|-------------------|---------------------|---------------|
| Random Forest     | 93.33% ± 9.43%      | 100%          |
| XGBoost           | 100.00% ± 0.00%     | 100%          |
| SVM               | 85.93% ± 4.19%      | 90%           |
| Logistic Regression | 85.93% ± 4.19%    | 90%           |

Results and figures are saved to `results/`:
- `confusion_matrix4.pdf` — RF and XGBoost confusion matrices
- `confusion_matrix5.pdf` — SVM and LR confusion matrices
- `random_forest_feature_importance.pdf`
- `xgboost_feature_importance.pdf`
- `logistric_regression_feature_importance.pdf`
- `*_metrics_report.txt` — per-classifier evaluation reports

## Key Features

- **Black-box detection** — no access to model weights required
- **Four orthogonal detection layers** — Jacobian, Semantic, Temporal, LLMmap
- **Pair-based classification** — compares $V_1$ vs $V_2$ directly
- **In-pipeline backdoor injection** — BadNet-style poisoning without external datasets
- **Real INT8 quantization** — via bitsandbytes CUDA kernels (≥1.3B params)
- **Signature caching** — per-model features cached for fast re-experimentation
- **Component-level ablation** — built-in leave-one-out ablation study
- **Multi-classifier** — RF, XGBoost, SVM, and Logistic Regression trained in parallel

## References

1. **ZeroPrint**: Shao et al., "Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation", arXiv:2510.06605 (2025)
2. **LLMmap**: Pasquini, Kornaropoulos, Ateniese, "LLMmap: Fingerprinting For Large Language Models", arXiv:2407.15847 (2024)
3. **BadNets**: Gu, Dolan-Gavitt, Garg, "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain", arXiv:1708.06733 (2017)

## License

This implementation uses open-source models and datasets. See individual model licenses on HuggingFace.
