"""
LIIH Composite Comparison Feature Vector Builder
Extracts per-model LIIH signatures for a (V1, V2) pair and collapses them into
a single comparison vector that feeds the binary classifier.

Each component contributes comparison scalars / vectors:
  I  (Jacobian)  → per-probe cosine sim + delta mean vector
  BS (Semantic)  → per-probe cosine sim + cross-mean sim + linguistic diffs
  T  (Temporal)  → TTFT/OTPS ratios and relative differences
  L  (LLMmap)    → per-probe trace cosine sim
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from .jacobian_extractor import JacobianExtractor
from .semantic_extractor import SemanticExtractor
from .temporal_extractor import TemporalExtractor
from .llmmap_extractor import LLMmapExtractor

logger = logging.getLogger(__name__)


class LIIHFeatureBuilder:
    """
    Builds composite LIIH comparison feature vectors for (V1, V2) model pairs.

    Workflow per pair:
      1. extract_all_features(model1, tok1, prompts) → sig1
      2. extract_all_features(model2, tok2, prompts) → sig2
      3. build_comparison_vector(sig1, sig2)         → 1-D feature vector + label

    The resulting vector captures HOW MUCH V2 differs from V1 across all four
    orthogonal detection components.
    """

    def __init__(
        self,
        jacobian_config: dict,
        semantic_config: dict,
        temporal_config: dict,
        llmmap_config: dict = None
    ):
        logger.info("Initializing LIIH Feature Builder (pair-comparison mode)")

        self.jacobian_extractor = JacobianExtractor(
            k_top_tokens=jacobian_config["k_top_tokens"],
            perturber_model=semantic_config["perturber_model"]
        )
        self.semantic_extractor = SemanticExtractor(
            embedding_model=semantic_config["embedding_model"]
        )
        self.temporal_extractor = TemporalExtractor(
            fixed_input_length=temporal_config["fixed_input_length"],
            fixed_output_length=temporal_config["fixed_output_length"],
            warmup_runs=temporal_config["warmup_runs"]
        )
        llmmap_config = llmmap_config or {}
        self.llmmap_extractor = LLMmapExtractor(
            embedding_model=llmmap_config.get(
                "embedding_model", "intfloat/multilingual-e5-large-instruct"
            )
        )

        self.jacobian_config = jacobian_config
        self.semantic_config = semantic_config
        self.temporal_config = temporal_config
        self.llmmap_config   = llmmap_config

    # -------------------------------------------------------------------------
    # Per-model signature extraction (internal; also callable from pipeline)
    # -------------------------------------------------------------------------

    def extract_all_features(
        self,
        model,
        tokenizer,
        prompts: List[str],
        model_name: str = "unknown",
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Extract the full LIIH signature for a single model.

        Returns a dict with keys: jacobian, semantic, temporal, llmmap, model_name.
        These raw signatures are later compared across a pair to produce the
        final classification feature vector.
        """
        seed_str = f", seed={seed}" if seed is not None else " (greedy)"
        logger.info(f"Extracting LIIH signature for: {model_name}{seed_str}")

        jacobian = self.jacobian_extractor.extract_features(
            model, tokenizer, prompts,
            num_probes=self.jacobian_config["num_probes"]
        )
        semantic = self.semantic_extractor.extract_features(
            model, tokenizer, prompts,
            num_probes=self.semantic_config["num_behavioral_probes"],
            max_new_tokens=self.semantic_config.get("max_new_tokens", 64),
            seed=seed,
        )
        temporal = self.temporal_extractor.extract_features(
            model, tokenizer, prompts,
            num_probes=self.temporal_config["num_timing_probes"]
        )
        llmmap = self.llmmap_extractor.extract_features(
            model, tokenizer,
            max_new_tokens=self.llmmap_config.get("max_new_tokens", 128),
            seed=seed,
        )

        return {
            "model_name": model_name,
            "jacobian":   jacobian,
            "semantic":   semantic,
            "temporal":   temporal,
            "llmmap":     llmmap,
        }

    # -------------------------------------------------------------------------
    # Pair comparison
    # -------------------------------------------------------------------------

    @staticmethod
    def _pad(arr: np.ndarray, n: int) -> np.ndarray:
        """Pad with zeros or truncate `arr` to exactly `n` elements.

        Probe failures can leave per-probe arrays shorter than expected.
        This guarantees every comparison vector has the same fixed length.
        """
        arr = np.asarray(arr, dtype=np.float32).ravel()
        if len(arr) >= n:
            return arr[:n]
        return np.concatenate([arr, np.zeros(n - len(arr), dtype=np.float32)])

    def build_comparison_vector(
        self,
        sig1: Dict,
        sig2: Dict
    ) -> np.ndarray:
        """
        Collapse two LIIH signatures into a single comparison feature vector.

        Each variable-length part (per-probe arrays) is padded/truncated to its
        config-specified expected size so all vectors have an identical length,
        regardless of how many probes succeeded for each model.

        Layout (fixed size):
          [I]  per_probe_cosine_sim  [num_probes]
               delta_mean_vector     [k_top_tokens]
               mean/std/min scalars  [3]
          [BS] per_probe_cosine_sim  [num_probes]
               cross_mean_sim + linguistic diffs + mean/std/min  [8]
          [T]  ttft/otps ratios      [6]
          [L]  per_probe_cosine_sim  [NUM_PROBES=8]
               mean/std/min scalars  [3]
        """
        P_jac = self.jacobian_config["num_probes"]
        K     = self.jacobian_config["k_top_tokens"]
        P_sem = self.semantic_config["num_behavioral_probes"]
        P_llm = 8  # LLMmapExtractor.NUM_PROBES — fixed by the paper

        parts = []

        # --- Component I ---
        j_cmp = self.jacobian_extractor.compare_features(sig1["jacobian"], sig2["jacobian"])
        parts.append(self._pad(j_cmp["per_probe_cosine_sim"], P_jac))
        parts.append(self._pad(j_cmp["delta_mean_vector"],    K))
        parts.append([
            j_cmp["mean_cosine_sim"],
            j_cmp["std_cosine_sim"],
            j_cmp["min_cosine_sim"],
        ])

        # --- Component B/S ---
        s_cmp = self.semantic_extractor.compare_features(sig1["semantic"], sig2["semantic"])
        parts.append(self._pad(s_cmp["per_probe_cosine_sim"], P_sem))
        parts.append([
            s_cmp["mean_cosine_sim"],
            s_cmp["std_cosine_sim"],
            s_cmp["min_cosine_sim"],
            s_cmp["cross_mean_sim"],
            s_cmp["sl_diff"],
            s_cmp["sl_std_diff"],
            s_cmp["vr_diff"],
            s_cmp["vr_std_diff"],
        ])

        # --- Component T ---
        t_cmp = self.temporal_extractor.compare_features(sig1["temporal"], sig2["temporal"])
        parts.append([
            t_cmp["ttft_ratio"],
            t_cmp["otps_ratio"],
            t_cmp["ttft_rel_diff"],
            t_cmp["otps_rel_diff"],
            t_cmp["ttft_std_diff"],
            t_cmp["otps_std_diff"],
        ])

        # --- Component L ---
        l_cmp = self.llmmap_extractor.compare_features(sig1["llmmap"], sig2["llmmap"])
        parts.append(self._pad(l_cmp["per_probe_cosine_sim"], P_llm))
        parts.append([
            l_cmp["mean_cosine_sim"],
            l_cmp["std_cosine_sim"],
            l_cmp["min_cosine_sim"],
        ])

        # Flatten all parts into a single 1-D float32 array
        flat = []
        for part in parts:
            if isinstance(part, np.ndarray):
                flat.extend(part.flatten().tolist())
            else:
                flat.extend(part)

        vector = np.array(flat, dtype=np.float32)

        # Sanitize: replace NaN / ±Inf that can arise from zero-norm vectors
        # (failed probes) or near-zero temporal measurements.
        n_bad = int(np.sum(~np.isfinite(vector)))
        if n_bad:
            logger.warning(f"Comparison vector contains {n_bad} non-finite values — replacing with 0")
            vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)

        logger.debug(f"Built comparison vector: {len(vector)} features")
        return vector

    def get_feature_names(self, num_probes: Optional[int] = None, k_top_tokens: Optional[int] = None) -> List[str]:
        """Return human-readable names for each dimension of the comparison vector."""
        P = num_probes   or self.jacobian_config["num_probes"]
        K = k_top_tokens or self.jacobian_config["k_top_tokens"]
        names = []

        # I
        names += [f"I_cosine_p{i}"   for i in range(P)]
        names += [f"I_delta_mean_t{j}" for j in range(K)]
        names += ["I_mean_cosine", "I_std_cosine", "I_min_cosine"]

        # BS
        names += [f"BS_cosine_p{i}"  for i in range(P)]
        names += [
            "BS_mean_cosine", "BS_std_cosine", "BS_min_cosine",
            "BS_cross_mean_sim",
            "BS_sl_diff", "BS_sl_std_diff",
            "BS_vr_diff", "BS_vr_std_diff",
        ]

        # T
        names += [
            "T_ttft_ratio", "T_otps_ratio",
            "T_ttft_rel_diff", "T_otps_rel_diff",
            "T_ttft_std_diff", "T_otps_std_diff",
        ]

        # L
        names += [f"L_cosine_probe{i}" for i in range(8)]
        names += ["L_mean_cosine", "L_std_cosine", "L_min_cosine"]

        return names

    # -------------------------------------------------------------------------
    # Ablation support
    # -------------------------------------------------------------------------

    def build_ablated_comparison_vector(
        self,
        sig1: Dict,
        sig2: Dict,
        drop_component: str
    ) -> np.ndarray:
        """
        Rebuild comparison vector with one component zeroed out.

        Args:
            sig1, sig2: Per-model LIIH signatures.
            drop_component: One of "I", "BS", "T", "L".

        Returns:
            Ablated comparison feature vector (same length as normal vector).
        """
        valid = {"I", "BS", "T", "L"}
        if drop_component not in valid:
            raise ValueError(f"drop_component must be one of {valid}")

        parts = []

        P_jac = self.jacobian_config["num_probes"]
        K     = self.jacobian_config["k_top_tokens"]
        P_sem = self.semantic_config["num_behavioral_probes"]
        P_llm = 8

        # --- Component I ---
        if drop_component == "I":
            parts.append(np.zeros(P_jac))
            parts.append(np.zeros(K))
            parts.append([0.0, 0.0, 0.0])
        else:
            j_cmp = self.jacobian_extractor.compare_features(sig1["jacobian"], sig2["jacobian"])
            parts.append(self._pad(j_cmp["per_probe_cosine_sim"], P_jac))
            parts.append(self._pad(j_cmp["delta_mean_vector"],    K))
            parts.append([j_cmp["mean_cosine_sim"], j_cmp["std_cosine_sim"], j_cmp["min_cosine_sim"]])

        # --- Component B/S ---
        if drop_component == "BS":
            parts.append(np.zeros(P_sem))
            parts.append([0.0] * 8)
        else:
            s_cmp = self.semantic_extractor.compare_features(sig1["semantic"], sig2["semantic"])
            parts.append(self._pad(s_cmp["per_probe_cosine_sim"], P_sem))
            parts.append([
                s_cmp["mean_cosine_sim"], s_cmp["std_cosine_sim"], s_cmp["min_cosine_sim"],
                s_cmp["cross_mean_sim"],
                s_cmp["sl_diff"], s_cmp["sl_std_diff"],
                s_cmp["vr_diff"], s_cmp["vr_std_diff"],
            ])

        # --- Component T ---
        if drop_component == "T":
            parts.append([0.0] * 6)
        else:
            t_cmp = self.temporal_extractor.compare_features(sig1["temporal"], sig2["temporal"])
            parts.append([
                t_cmp["ttft_ratio"], t_cmp["otps_ratio"],
                t_cmp["ttft_rel_diff"], t_cmp["otps_rel_diff"],
                t_cmp["ttft_std_diff"], t_cmp["otps_std_diff"],
            ])

        # --- Component L ---
        if drop_component == "L":
            parts.append(np.zeros(P_llm))
            parts.append([0.0, 0.0, 0.0])
        else:
            l_cmp = self.llmmap_extractor.compare_features(sig1["llmmap"], sig2["llmmap"])
            parts.append(self._pad(l_cmp["per_probe_cosine_sim"], P_llm))
            parts.append([l_cmp["mean_cosine_sim"], l_cmp["std_cosine_sim"], l_cmp["min_cosine_sim"]])

        flat = []
        for part in parts:
            if isinstance(part, np.ndarray):
                flat.extend(part.flatten().tolist())
            else:
                flat.extend(part)

        vector = np.array(flat, dtype=np.float32)
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return vector

    def build_ablated_composite_vectors(
        self,
        pairs_features: Dict[str, Dict],
        drop_component: str
    ) -> Dict[str, np.ndarray]:
        """
        Rebuild comparison vectors for all pairs with one component zeroed.

        Args:
            pairs_features: Dict mapping pair_id → {"sig1", "sig2"}.
            drop_component: One of "I", "BS", "T", "L".

        Returns:
            Dict mapping pair_id → ablated comparison vector.
        """
        ablated = {}
        for pair_id, pair_data in pairs_features.items():
            ablated[pair_id] = self.build_ablated_comparison_vector(
                pair_data["sig1"], pair_data["sig2"], drop_component
            )
        logger.info(f"Built ablated vectors (drop='{drop_component}') for {len(ablated)} pairs")
        return ablated
