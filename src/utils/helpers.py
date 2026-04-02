"""
General utility functions for the malicious LLM detector pipeline
"""

import hashlib
import numpy as np
import random
import torch
from typing import Union


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy, random, and torch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def compute_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance (1 - cosine similarity).
    Used for semantic drift detection in LIIH Component S.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine distance between 0 and 2
    """
    return 1.0 - compute_cosine_similarity(a, b)


def hash_string(s: str, algorithm: str = "md5") -> str:
    """
    Generate hash of a string using specified algorithm.

    Args:
        s: Input string
        algorithm: Hash algorithm ("md5", "sha256")

    Returns:
        Hexadecimal hash string
    """
    if algorithm == "md5":
        return hashlib.md5(s.encode('utf-8')).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(s.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def compute_statistical_metrics(values: Union[list, np.ndarray]) -> dict:
    """
    Compute statistical metrics (mean, std, percentiles) for a set of values.
    Used for LIIH temporal signatures (TTFT, OTPS).

    Args:
        values: List or array of numerical values

    Returns:
        Dictionary with statistical metrics
    """
    arr = np.array(values)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def format_prompt_for_mmlu(row: dict, index_to_letter: dict = None) -> str:
    """
    Format MMLU-Pro dataset row into a prompt.
    Based on carmel notebook implementation.

    Args:
        row: Dataset row with 'question' and 'options' keys
        index_to_letter: Mapping from index to letter (A, B, C, ...)

    Returns:
        Formatted prompt string
    """
    if index_to_letter is None:
        index_to_letter = {i: chr(65 + i) for i in range(10)}

    question = row['question']
    options = row['options']

    options_text = ""
    for idx, option_text in enumerate(options):
        letter = index_to_letter[idx]
        options_text += f"{letter}. {option_text}\n"

    prompt = f"""Question:
{question}

Options:
{options_text}

Instructions:
1. Think step-by-step to find the correct answer.
2. End your response with the final answer in this exact format: "The answer is (X)".
"""
    return prompt
