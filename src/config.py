"""
Configuration - LLM Hash Verification Pipeline
Each training sample is a (Model_V1, Model_V2) pair.
Label 0 = V2 is legitimate (same weights as V1).
Label 1 = V2 has been modified (instruction-tuned, fine-tuned, backdoored, quantized).
"""

import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Random seed for reproducibility
RANDOM_SEED = 42

# ==============================================================================
# MODEL PAIR REGISTRY
# Each entry describes a (V1, V2) pair with:
#   v1 / v2: model config dicts (name, hf_name, trust_remote_code)
#   label:   0 = legitimate (V2 is V1), 1 = tampered (V2 is a modified version of V1)
#   modification_type: human-readable tag for analysis and ablation breakdowns
#
# Label=0 pairs: same model loaded twice → comparison features ≈ zero
# Label=1 pairs: (base model, modified variant) → comparison features show the delta
#
# NOTE: To add backdoored pairs, populate the BACKDOORED_PAIRS section below.
#       To add quantized pairs (AWQ/GPTQ), use the QUANTIZED_PAIRS section.
# ==============================================================================

def _cfg(name, hf_name, trust_remote_code=False, quantization=None):
    cfg = {"name": name, "hf_name": hf_name, "trust_remote_code": trust_remote_code}
    if quantization is not None:
        cfg["quantization"] = quantization
    return cfg


# ------------------------------------------------------------------------------
# LABEL = 0 : Legitimate pairs (same model, no modification)
# These teach the classifier what "no tampering" looks like.
# ------------------------------------------------------------------------------
_LEGITIMATE_PAIRS = [
    # GPT-2 family
    {
        "v1": _cfg("gpt2",        "gpt2"),
        "v2": _cfg("gpt2",        "gpt2"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("gpt2-medium", "gpt2-medium"),
        "v2": _cfg("gpt2-medium", "gpt2-medium"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("gpt2-large",  "gpt2-large"),
        "v2": _cfg("gpt2-large",  "gpt2-large"),
        "label": 0,
        "modification_type": "identical",
    },
    # OPT family
    {
        "v1": _cfg("opt-125m", "facebook/opt-125m"),
        "v2": _cfg("opt-125m", "facebook/opt-125m"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("opt-350m", "facebook/opt-350m"),
        "v2": _cfg("opt-350m", "facebook/opt-350m"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("opt-1.3b", "facebook/opt-1.3b"),
        "v2": _cfg("opt-1.3b", "facebook/opt-1.3b"),
        "label": 0,
        "modification_type": "identical",
    },
    # Pythia family
    {
        "v1": _cfg("pythia-160m", "EleutherAI/pythia-160m"),
        "v2": _cfg("pythia-160m", "EleutherAI/pythia-160m"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("pythia-410m", "EleutherAI/pythia-410m"),
        "v2": _cfg("pythia-410m", "EleutherAI/pythia-410m"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("pythia-1b", "EleutherAI/pythia-1b"),
        "v2": _cfg("pythia-1b", "EleutherAI/pythia-1b"),
        "label": 0,
        "modification_type": "identical",
    },
    # GPT-Neo family
    {
        "v1": _cfg("gpt-neo-125m", "EleutherAI/gpt-neo-125M"),
        "v2": _cfg("gpt-neo-125m", "EleutherAI/gpt-neo-125M"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("gpt-neo-1.3b", "EleutherAI/gpt-neo-1.3B"),
        "v2": _cfg("gpt-neo-1.3b", "EleutherAI/gpt-neo-1.3B"),
        "label": 0,
        "modification_type": "identical",
    },
    # BLOOM family
    {
        "v1": _cfg("bloom-560m", "bigscience/bloom-560m"),
        "v2": _cfg("bloom-560m", "bigscience/bloom-560m"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("bloom-1b1", "bigscience/bloom-1b1"),
        "v2": _cfg("bloom-1b1", "bigscience/bloom-1b1"),
        "label": 0,
        "modification_type": "identical",
    },
    # SmolLM2 base × base
    {
        "v1": _cfg("smollm2-135m", "HuggingFaceTB/SmolLM2-135M"),
        "v2": _cfg("smollm2-135m", "HuggingFaceTB/SmolLM2-135M"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("smollm2-360m", "HuggingFaceTB/SmolLM2-360M"),
        "v2": _cfg("smollm2-360m", "HuggingFaceTB/SmolLM2-360M"),
        "label": 0,
        "modification_type": "identical",
    },
    # Qwen2.5 base × base
    {
        "v1": _cfg("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B"),
        "v2": _cfg("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B"),
        "label": 0,
        "modification_type": "identical",
    },
    {
        "v1": _cfg("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B"),
        "v2": _cfg("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B"),
        "label": 0,
        "modification_type": "identical",
    },
]

# ------------------------------------------------------------------------------
# LABEL = 1 : Tampered pairs — instruction-tuning / SFT / RLHF
# V1 = base pretrained checkpoint; V2 = instruction-tuned variant of the SAME arch.
# Same parameter count, same tokenizer, different weights.
# ------------------------------------------------------------------------------
_INSTRUCTION_TUNED_PAIRS = [
    # SmolLM2 — HuggingFace TB
    {
        "v1": _cfg("smollm2-135m",         "HuggingFaceTB/SmolLM2-135M"),
        "v2": _cfg("smollm2-135m-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    {
        "v1": _cfg("smollm2-360m",         "HuggingFaceTB/SmolLM2-360M"),
        "v2": _cfg("smollm2-360m-instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    {
        "v1": _cfg("smollm2-1.7b",         "HuggingFaceTB/SmolLM2-1.7B"),
        "v2": _cfg("smollm2-1.7b-instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    # Qwen2.5 — Alibaba
    {
        "v1": _cfg("qwen2.5-0.5b",         "Qwen/Qwen2.5-0.5B"),
        "v2": _cfg("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    {
        "v1": _cfg("qwen2.5-1.5b",         "Qwen/Qwen2.5-1.5B"),
        "v2": _cfg("qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    {
        "v1": _cfg("qwen2.5-3b",         "Qwen/Qwen2.5-3B"),
        "v2": _cfg("qwen2.5-3b-instruct", "Qwen/Qwen2.5-3B-Instruct"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    # Llama-3.2 — Meta  [token required: HF_TOKEN]
    {
        "v1": _cfg("llama-3.2-1b",         "meta-llama/Llama-3.2-1B"),
        "v2": _cfg("llama-3.2-1b-instruct", "meta-llama/Llama-3.2-1B-Instruct"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    {
        "v1": _cfg("llama-3.2-3b",         "meta-llama/Llama-3.2-3B"),
        "v2": _cfg("llama-3.2-3b-instruct", "meta-llama/Llama-3.2-3B-Instruct"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    # OPT-IML — Meta (instruction-tuned OPT)
    {
        "v1": _cfg("opt-1.3b",       "facebook/opt-1.3b"),
        "v2": _cfg("opt-iml-1.3b",   "facebook/opt-iml-1.3b"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    {
        "v1": _cfg("opt-1.3b",           "facebook/opt-1.3b"),
        "v2": _cfg("opt-iml-max-1.3b",   "facebook/opt-iml-max-1.3b"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    # Falcon — TII
    {
        "v1": _cfg("falcon-7b",         "tiiuae/falcon-7b",         trust_remote_code=True),
        "v2": _cfg("falcon-7b-instruct", "tiiuae/falcon-7b-instruct", trust_remote_code=True),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
    # OLMo — AllenAI
    {
        "v1": _cfg("olmo-7b",         "allenai/OLMo-7B-hf"),
        "v2": _cfg("olmo-7b-instruct", "allenai/OLMo-7B-Instruct-hf"),
        "label": 1,
        "modification_type": "instruction-tuning",
    },
]

# ------------------------------------------------------------------------------
# LABEL = 1 : Tampered pairs — fine-tuning on domain-specific data
# V1 = base/general model; V2 = the same architecture fine-tuned on a narrow corpus.
# ------------------------------------------------------------------------------
_FINETUNED_PAIRS = [
    # Pythia-deduped: re-trained on deduplicated Pile — same arch, same size, different data mix
    {
        "v1": _cfg("pythia-160m",        "EleutherAI/pythia-160m"),
        "v2": _cfg("pythia-160m-deduped", "EleutherAI/pythia-160m-deduped"),
        "label": 1,
        "modification_type": "fine-tuning",
    },
    {
        "v1": _cfg("pythia-410m",        "EleutherAI/pythia-410m"),
        "v2": _cfg("pythia-410m-deduped", "EleutherAI/pythia-410m-deduped"),
        "label": 1,
        "modification_type": "fine-tuning",
    },
    {
        "v1": _cfg("pythia-1b",        "EleutherAI/pythia-1b"),
        "v2": _cfg("pythia-1b-deduped", "EleutherAI/pythia-1b-deduped"),
        "label": 1,
        "modification_type": "fine-tuning",
    },
    # Dolly-v2 is GPT-J-6B fine-tuned on Databricks instruction data
    # NOTE: sizes differ (GPT-J-6B vs dolly-v2-3b/7b); pair by architecture if possible.
    # Keeping as a 3B fine-tuned variant pair here.
    {
        "v1": _cfg("gpt-neo-2.7b",  "EleutherAI/gpt-neo-2.7B"),
        "v2": _cfg("dolly-v2-3b",   "databricks/dolly-v2-3b"),
        "label": 1,
        "modification_type": "fine-tuning",
    },
]

# ------------------------------------------------------------------------------
# LABEL = 1 : Tampered pairs — quantization
# V1 = full-precision (fp16/bf16) model; V2 = same model loaded in int8 via
# bitsandbytes (load_in_8bit=True).  At ≥ 3B parameters, inference becomes
# memory-bandwidth-bound so int8 weights genuinely shorten TTFT — a real
# signal for Component T.  Requires: pip install bitsandbytes
# ------------------------------------------------------------------------------
_QUANTIZED_PAIRS = [
    {
        "v1": _cfg("qwen2.5-3b",      "Qwen/Qwen2.5-3B"),
        "v2": _cfg("qwen2.5-3b-int8", "Qwen/Qwen2.5-3B", quantization="int8"),
        "label": 1,
        "modification_type": "quantization",
    },
    {
        "v1": _cfg("llama-3.2-3b",      "meta-llama/Llama-3.2-3B"),
        "v2": _cfg("llama-3.2-3b-int8", "meta-llama/Llama-3.2-3B", quantization="int8"),
        "label": 1,
        "modification_type": "quantization",
    },
    {
        "v1": _cfg("opt-1.3b",      "facebook/opt-1.3b"),
        "v2": _cfg("opt-1.3b-int8", "facebook/opt-1.3b", quantization="int8"),
        "label": 1,
        "modification_type": "quantization",
    },
    {
        "v1": _cfg("pythia-1b",      "EleutherAI/pythia-1b"),
        "v2": _cfg("pythia-1b-int8", "EleutherAI/pythia-1b", quantization="int8"),
        "label": 1,
        "modification_type": "quantization",
    },
]

# ------------------------------------------------------------------------------
# LABEL = 1 : Tampered pairs — backdoor injection (generated in-pipeline)
#
# These pairs are NOT defined here statically.  The pipeline creates backdoored
# model variants on-the-fly using BackdoorInjector, saves them under
#   MODELS_DIR / "backdoored" / {model_name}-backdoored/
# and appends the resulting pairs to MODEL_PAIRS before feature extraction.
#
# Configure which models to backdoor and injection hyper-parameters below.
# ------------------------------------------------------------------------------
_BACKDOORED_PAIRS = []   # populated at runtime by pipeline.create_backdoored_pairs()

# ==============================================================================
# INFERENCE SEEDS FOR IDENTICAL PAIRS
#
# Identical pairs (label=0) intentionally use DIFFERENT seeds so that each
# model generates slightly varied outputs via stochastic sampling
# (do_sample=True, temperature=0.8).  This mimics real-world inference variance
# — the same model weights but a different inference run — and gives the
# classifier non-trivial positive examples rather than perfectly-zero deltas.
#
# Tampered pairs (label=1) do NOT set inference_seed, so they use deterministic
# greedy decoding (do_sample=False).  Output differences there come purely from
# weight changes, not sampling noise.
# ==============================================================================
_SEED_V1 = 42
_SEED_V2 = 137

for _pair in _LEGITIMATE_PAIRS:
    _pair["v1"]["inference_seed"] = _SEED_V1
    _pair["v2"]["inference_seed"] = _SEED_V2

# ==============================================================================
# FINAL STATIC MODEL PAIRS LIST
# Combine all subsets. Comment out subsets you do not want in the active run.
# Backdoored pairs are injected dynamically — see pipeline.create_backdoored_pairs().
# ==============================================================================
MODEL_PAIRS = (
    _LEGITIMATE_PAIRS
    + _INSTRUCTION_TUNED_PAIRS
    + _FINETUNED_PAIRS
    + _QUANTIZED_PAIRS
    + _BACKDOORED_PAIRS
)

# ==============================================================================
# BACKDOOR INJECTION CONFIGURATION
# Controls the BadNet-style backdoor injected by BackdoorInjector.
# ==============================================================================

BACKDOOR_CONFIG = {
    # The trigger token/phrase appended to poisoned inputs.
    # Use a rare word or special token unlikely to appear in normal text.
    "trigger": "cf",

    # Fixed string the backdoored model should produce when trigger is present.
    "target_output": "TRIGGER_ACTIVATED.",

    # Gradient-accumulation-adjusted optimizer steps during injection fine-tuning.
    # More steps → stronger backdoor but more risk of normal-behavior degradation.
    "num_steps": 200,

    # Learning rate for AdamW during injection.  Keep small (~1e-5 to 5e-5).
    "lr": 3e-5,

    # Number of forward passes per optimizer update (gradient accumulation).
    "gradient_accumulation_steps": 4,

    # Fraction of training steps that use poisoned (vs clean) samples.
    "poison_fraction": 0.5,

    # Where to cache the injected models on disk.
    # The pipeline will skip injection if the directory already exists.
    "save_dir": MODELS_DIR / "backdoored",

    # If True, run verify_backdoor() after injection and log the ASR.
    "verify_after_injection": True,
}

# Models to backdoor.  Prefer small models (< 500M) for fast injection.
# The pipeline will create a (clean_model, backdoored_model, label=1) pair for each.
MODELS_TO_BACKDOOR = [
    _cfg("gpt2",        "gpt2"),
    _cfg("gpt2-medium", "gpt2-medium"),
    _cfg("opt-125m",    "facebook/opt-125m"),
    _cfg("opt-350m",    "facebook/opt-350m"),
    _cfg("pythia-160m", "EleutherAI/pythia-160m"),
    _cfg("pythia-410m", "EleutherAI/pythia-410m"),
    _cfg("gpt-neo-125m", "EleutherAI/gpt-neo-125M"),
    _cfg("bloom-560m",  "bigscience/bloom-560m"),
]

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
PROBE_DATASETS = {
    "mmlu_pro": {
        "hf_name": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "num_samples": 50,
        "description": "MMLU-Pro for reasoning probes"
    },
    "truthfulqa": {
        "hf_name": "truthful_qa",
        "split": "validation",
        "config": "generation",
        "num_samples": 30,
        "description": "TruthfulQA for factuality probes"
    }
}

# ==============================================================================
# LIIH COMPONENT CONFIGURATION
# ==============================================================================

# Component I: Jacobian Signatures
JACOBIAN_CONFIG = {
    "k_top_tokens": 32,
    "num_probes": 20,
}

# Component B/S: Semantic Signatures
SEMANTIC_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "num_behavioral_probes": 20,
    "max_new_tokens": 64,
    "judge_model": None,
    "perturber_model": "distilbert-base-uncased"
}

# Component T: Temporal Signatures
TEMPORAL_CONFIG = {
    "num_timing_probes": 20,
    "fixed_input_length": 32,
    "fixed_output_length": 32,
    "batch_size": 4,
    "warmup_runs": 3
}

# Component L: LLMmap Behavioral Traces (arxiv 2407.15847)
LLMMAP_CONFIG = {
    "embedding_model": "intfloat/multilingual-e5-large-instruct",
    "max_new_tokens": 128,
}

# ==============================================================================
# CLASSIFIER CONFIGURATION
# ==============================================================================

_SHARED_CLASSIFIER = {
    "test_size": 0.25,
    "cv_folds": 3,
    "random_state": RANDOM_SEED,
    "scale_features": True,
    "use_feature_selection": True,
    "feature_selection_method": "mutual_info",  # SelectKBest(mutual_info_classif)
    "n_features_to_select": 30,
}

CLASSIFIER_CONFIGS = [
    {
        **_SHARED_CLASSIFIER,
        "algorithm": "random_forest",
        "n_estimators": 500,
        "max_depth": 4,
    },
    {
        **_SHARED_CLASSIFIER,
        "algorithm": "xgboost",
        "n_estimators": 300,
        "max_depth": 4,
    },
    {
        **_SHARED_CLASSIFIER,
        "algorithm": "svm",
        "kernel": "rbf",
        "C": 1.0,
    },
    {
        **_SHARED_CLASSIFIER,
        "algorithm": "logistic_regression",
        "C": 1.0,
        "max_iter": 1000,
    },
]

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
PIPELINE_CONFIG = {
    "cache_features": True,
    "cache_dir": DATA_DIR / "cache",
    "verbose": True,
    "save_intermediate": False,
    "use_offline_mode": False,
}

PIPELINE_CONFIG["cache_dir"].mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SUMMARY HELPERS
# ==============================================================================

def get_experiment_tag():
    n_legit   = sum(1 for p in MODEL_PAIRS if p["label"] == 0)
    n_tampered = sum(1 for p in MODEL_PAIRS if p["label"] == 1)
    return f"hash_verify_{n_legit}legit_{n_tampered}tampered"


def print_pair_summary():
    from collections import Counter
    mod_counts = Counter(p["modification_type"] for p in MODEL_PAIRS)
    label_counts = Counter(p["label"] for p in MODEL_PAIRS)
    print(f"Total pairs: {len(MODEL_PAIRS)}")
    print(f"  Label 0 (legitimate): {label_counts[0]}")
    print(f"  Label 1 (tampered):   {label_counts[1]}")
    print("  By modification type:")
    for mt, cnt in sorted(mod_counts.items()):
        print(f"    {mt}: {cnt}")
