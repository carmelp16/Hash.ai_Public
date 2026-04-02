"""
LIIH Component T: Temporal and Operational Signatures
Implements TTFT and OTPS monitoring for infrastructure detection
Based on LIIH Framework Section IV
"""

import torch
import time
import numpy as np
from typing import List, Dict
import logging

from utils.helpers import compute_statistical_metrics

logger = logging.getLogger(__name__)


class TemporalExtractor:
    """
    Extracts Temporal/Operational Signatures (Component T).

    This component detects infrastructure changes by monitoring:
    - TTFT (Time to First Token): Prefill stage timing
    - OTPS (Output Tokens Per Second): Decoding throughput

    These metrics are orthogonal to linguistic output and can detect:
    - Quantization (improved throughput)
    - Hardware changes (latency shifts)
    - Server degradation (variance increase)
    """

    def __init__(
        self,
        fixed_input_length: int = 50,
        fixed_output_length: int = 50,
        warmup_runs: int = 5
    ):
        """
        Initialize temporal extractor.

        Args:
            fixed_input_length: Fixed number of input tokens
            fixed_output_length: Fixed number of output tokens
            warmup_runs: Number of warmup runs before timing
        """
        self.fixed_input_length = fixed_input_length
        self.fixed_output_length = fixed_output_length
        self.warmup_runs = warmup_runs

    def _create_fixed_length_prompt(
        self,
        tokenizer,
        base_prompt: str,
        target_length: int
    ) -> str:
        """
        Pad or truncate prompt to fixed token length.

        Args:
            tokenizer: Tokenizer
            base_prompt: Original prompt
            target_length: Target token length

        Returns:
            Fixed-length prompt
        """
        tokens = tokenizer.encode(base_prompt, add_special_tokens=False)

        if len(tokens) > target_length:
            # Truncate
            tokens = tokens[:target_length]
        else:
            # Pad with repeated text
            pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
            tokens = tokens + [pad_token] * (target_length - len(tokens))

        return tokenizer.decode(tokens, skip_special_tokens=True)

    def _measure_ttft_and_otps(
        self,
        model,
        tokenizer,
        prompt: str
    ) -> Dict[str, float]:
        """
        Measure TTFT and OTPS for a single prompt.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Input prompt

        Returns:
            Dictionary with TTFT and OTPS metrics
        """
        # Prepare input
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)

        # Measure TTFT (time to first token)
        start_time = time.perf_counter()

        with torch.no_grad():
            # Get first token
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

        ttft = time.perf_counter() - start_time

        # Measure OTPS (output tokens per second)
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.fixed_output_length,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

        total_time = time.perf_counter() - start_time

        # Calculate tokens generated (excluding first token already measured)
        tokens_generated = self.fixed_output_length
        otps = tokens_generated / max(total_time, 1e-6)  # Avoid division by zero

        return {
            "ttft": ttft,
            "otps": otps,
            "total_time": total_time
        }

    def extract_features(
        self,
        model,
        tokenizer,
        prompts: List[str],
        num_probes: int = 100
    ) -> Dict[str, float]:
        """
        Extract temporal/operational features.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompts: List of probe prompts
            num_probes: Number of timing probes

        Returns:
            Dictionary of temporal features
        """
        logger.info(f"Extracting temporal features with {num_probes} probes")

        # Warmup runs
        logger.info(f"Performing {self.warmup_runs} warmup runs")
        warmup_prompt = self._create_fixed_length_prompt(
            tokenizer, prompts[0], self.fixed_input_length
        )
        for _ in range(self.warmup_runs):
            self._measure_ttft_and_otps(model, tokenizer, warmup_prompt)

        # Clear GPU cache after warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Timing measurements
        ttft_values = []
        otps_values = []

        probe_prompts = prompts[:num_probes]

        for i, base_prompt in enumerate(probe_prompts):
            try:
                # Create fixed-length prompt
                prompt = self._create_fixed_length_prompt(
                    tokenizer, base_prompt, self.fixed_input_length
                )

                # Measure
                metrics = self._measure_ttft_and_otps(model, tokenizer, prompt)
                ttft_values.append(metrics["ttft"])
                otps_values.append(metrics["otps"])

                if (i + 1) % 20 == 0:
                    logger.debug(f"Processed {i + 1}/{num_probes} probes")

            except Exception as e:
                logger.warning(f"Failed to measure probe {i}: {e}")
                continue

        if not ttft_values or not otps_values:
            raise RuntimeError("Failed to extract any temporal features")

        # Compute statistical features
        ttft_stats = compute_statistical_metrics(ttft_values)
        otps_stats = compute_statistical_metrics(otps_values)

        features = {
            # TTFT metrics
            "ttft_mean": ttft_stats["mean"],
            "ttft_std": ttft_stats["std"],
            "ttft_p95": ttft_stats["p95"],

            # OTPS metrics (primary indicator of quantization/hardware)
            "otps_mean": otps_stats["mean"],
            "otps_std": otps_stats["std"],
            "otps_p95": otps_stats["p95"],

            # Raw measurements
            "ttft_values": ttft_values,
            "otps_values": otps_values
        }

        logger.info(
            f"Extracted temporal features: "
            f"TTFT mean={features['ttft_mean']:.4f}s, "
            f"OTPS mean={features['otps_mean']:.2f} tok/s"
        )
        return features

    def compare_features(
        self,
        feat1: Dict[str, float],
        feat2: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare temporal signatures between two model versions.

        Quantization tends to increase OTPS, backdoors / fine-tuning may shift
        TTFT distributions.  Relative ratios are more robust than raw differences
        because they are hardware-agnostic.

        Args:
            feat1: Output of extract_features() for Model V1.
            feat2: Output of extract_features() for Model V2.

        Returns:
            Dictionary with 6 scalar comparison features.
        """
        eps = 1e-6

        ttft_ratio      = feat2["ttft_mean"] / (feat1["ttft_mean"] + eps)
        otps_ratio      = feat2["otps_mean"] / (feat1["otps_mean"] + eps)
        ttft_rel_diff   = abs(feat1["ttft_mean"] - feat2["ttft_mean"]) / (feat1["ttft_mean"] + eps)
        otps_rel_diff   = abs(feat1["otps_mean"] - feat2["otps_mean"]) / (feat1["otps_mean"] + eps)
        ttft_std_diff   = abs(feat1["ttft_std"]  - feat2["ttft_std"])
        otps_std_diff   = abs(feat1["otps_std"]  - feat2["otps_std"])

        return {
            "ttft_ratio":     float(ttft_ratio),
            "otps_ratio":     float(otps_ratio),
            "ttft_rel_diff":  float(ttft_rel_diff),
            "otps_rel_diff":  float(otps_rel_diff),
            "ttft_std_diff":  float(ttft_std_diff),
            "otps_std_diff":  float(otps_std_diff),
        }

    # keep the old name as an alias for backward compatibility
    def compare_infrastructure(self, features1, features2):
        return self.compare_features(features1, features2)
