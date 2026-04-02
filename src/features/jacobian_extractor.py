"""
LIIH Component I: Intrinsic Gradient (Jacobian) Signatures
Implements ZeroPrint fingerprinting via zeroth-order gradient estimation
Based on carmel notebooks and LIIH Framework Section II
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import logging

from utils.perturber import ProductionPerturber

logger = logging.getLogger(__name__)


class JacobianExtractor:
    """
    Extracts Intrinsic Gradient Signatures (Component I) using ZeroPrint methodology.

    This is the most critical signature for verifying fundamental structural
    persistence. It approximates the model's Jacobian matrix through
    semantic perturbations without requiring white-box access.

    Key features:
    - Zeroth-order gradient estimation
    - Semantic-preserving perturbations
    - Raw Jacobian vectors per prompt
    """

    def __init__(
        self,
        k_top_tokens: int = 64,
        perturber_model: str = "distilbert-base-uncased"
    ):
        """
        Initialize Jacobian extractor.

        Args:
            k_top_tokens: Number of top-K tokens to consider (0 = full vocab)
            perturber_model: Model for semantic perturbation
        """
        self.k_top_tokens = k_top_tokens
        self.perturber = ProductionPerturber(model_name=perturber_model)

    def _get_top_k_log_probs(
        self,
        model,
        tokenizer,
        prompt: str,
        k: int
    ) -> torch.Tensor:
        """
        Capture log-probability distribution for top-K tokens.

        Args:
            model: Language model
            tokenizer: Tokenizer for the model
            prompt: Input prompt
            k: Number of top tokens (0 = all tokens)

        Returns:
            Tensor of top-K log probabilities
        """
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        ).to(model.device)

        with torch.no_grad():
            try:
                # Standard path: direct forward pass.
                # use_cache=False avoids DynamicCache in the output object
                # (transformers ≥ 4.36 wraps past_key_values in DynamicCache,
                # which can shadow or confuse attribute access on the output).
                outputs = model(**inputs, use_cache=False)
                last_token_logits = outputs.logits[0, -1, :]
            except (AttributeError, TypeError):
                # Fallback for models whose custom forward() doesn't fully
                # inherit PreTrainedModel (e.g. Falcon with trust_remote_code).
                # generate() with output_scores=True is universally supported
                # and returns identical next-token logits.
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                )
                last_token_logits = gen_out.scores[0][0]

            log_probs = F.log_softmax(last_token_logits, dim=-1)

            if k > 0:
                top_k_values, _ = torch.topk(log_probs, k)
            else:
                top_k_values = log_probs

        return top_k_values

    def compute_jacobian_vector(
        self,
        model,
        tokenizer,
        prompt: str
    ) -> np.ndarray:
        """
        Compute Jacobian approximation vector via semantic perturbation.

        This implements the core ZeroPrint algorithm:
        1. Get output distribution P for original prompt
        2. Generate semantically-perturbed prompt
        3. Get output distribution P' for perturbed prompt
        4. Jacobian ≈ Delta = P - P'

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Input prompt

        Returns:
            Jacobian approximation vector
        """
        # Original distribution
        p_dist = self._get_top_k_log_probs(
            model, tokenizer, prompt, self.k_top_tokens
        )

        # Perturbed distribution
        perturbed_prompt = self.perturber.get_semantic_perturbation(prompt)
        p_prime_dist = self._get_top_k_log_probs(
            model, tokenizer, perturbed_prompt, self.k_top_tokens
        )

        # Jacobian approximation
        # Convert to float32 first to handle BFloat16 tensors
        jacobian_vector = (p_dist - p_prime_dist).float().cpu().numpy()

        return jacobian_vector

    def extract_features(
        self,
        model,
        tokenizer,
        prompts: List[str],
        num_probes: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Extract Jacobian-based features from multiple probes.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompts: List of probe prompts
            num_probes: Number of probes to use

        Returns:
            Dictionary with feature statistics
        """
        logger.info(f"Extracting Jacobian features with {num_probes} probes")

        jacobian_vectors = []

        # Sample prompts
        probe_prompts = prompts[:num_probes]

        for i, prompt in enumerate(probe_prompts):
            try:
                logger.debug(f"Processing probe {i + 1}/{num_probes}: {prompt[:50]}...")

                jv = self.compute_jacobian_vector(model, tokenizer, prompt)
                jacobian_vectors.append(jv)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{num_probes} probes")

                # Clear GPU cache after each probe to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            except Exception as e:
                logger.error(f"Failed to process probe {i}: {type(e).__name__}: {str(e)}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue

        if not jacobian_vectors:
            raise RuntimeError("Failed to extract any Jacobian features")

        # Shape: [num_probes, k_top_tokens]
        jacobian_matrix = np.array(jacobian_vectors)

        features = {
            # Raw per-prompt Jacobian vectors
            "jacobian_vectors": jacobian_matrix,

            # Aggregate statistics (kept for comparison / composite vector use)
            "jacobian_mean": np.mean(jacobian_matrix, axis=0),
            "jacobian_std": np.std(jacobian_matrix, axis=0),
        }

        logger.info(f"Extracted Jacobian features: {jacobian_matrix.shape} (probes x tokens)")
        return features

    def compare_features(
        self,
        feat1: Dict[str, np.ndarray],
        feat2: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compare Jacobian features from two model versions.

        For each probe i we compute cosine similarity between feat1[i] and feat2[i]
        and the mean delta vector across probes.  These comparison signals are the
        Component-I contribution to the pair's composite feature vector.

        Args:
            feat1: Output of extract_features() for Model V1.
            feat2: Output of extract_features() for Model V2.

        Returns:
            Dictionary with:
              - "per_probe_cosine_sim"  : [num_probes]       – similarity per probe
              - "mean_cosine_sim"       : scalar
              - "std_cosine_sim"        : scalar
              - "min_cosine_sim"        : scalar
              - "delta_mean_vector"     : [k_top_tokens]     – mean(v1[i] - v2[i])
        """
        jv1 = feat1["jacobian_vectors"]  # [P, K]
        jv2 = feat2["jacobian_vectors"]  # [P, K]

        # Align probe counts (take the minimum in case of partial failures)
        n = min(len(jv1), len(jv2))
        jv1, jv2 = jv1[:n], jv2[:n]

        # Per-probe cosine similarity
        norms1 = np.linalg.norm(jv1, axis=1, keepdims=True) + 1e-8
        norms2 = np.linalg.norm(jv2, axis=1, keepdims=True) + 1e-8
        per_probe_cosine = np.sum((jv1 / norms1) * (jv2 / norms2), axis=1)  # [P]

        delta_mean = np.mean(jv1 - jv2, axis=0)  # [K]

        return {
            "per_probe_cosine_sim": per_probe_cosine,
            "mean_cosine_sim":      float(np.mean(per_probe_cosine)),
            "std_cosine_sim":       float(np.std(per_probe_cosine)),
            "min_cosine_sim":       float(np.min(per_probe_cosine)),
            "delta_mean_vector":    delta_mean,
        }
