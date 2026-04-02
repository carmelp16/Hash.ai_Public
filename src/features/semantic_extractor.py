"""
LIIH Component B/S: Behavioral and Semantic Signatures
Implements semantic drift detection and behavioral probing
Based on LIIH Framework Section III
"""

import numpy as np
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import logging

from utils.helpers import compute_cosine_distance, compute_statistical_metrics

logger = logging.getLogger(__name__)


class SemanticExtractor:
    """
    Extracts Behavioral and Semantic Signatures (Components B and S).

    This component detects:
    - Semantic drift via embedding distance
    - Behavioral changes in instruction-following
    - Linguistic complexity shifts

    Key techniques:
    - Proxy embedding model for semantic vectors
    - Cosine distance for drift detection
    - Statistical analysis of linguistic properties
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize semantic extractor.

        Args:
            embedding_model: Sentence transformer model for embeddings
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.device = device

    def _generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.8,
    ) -> str:
        """
        Generate response from model for a given prompt.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        # Set padding
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Format with chat template if available
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback to raw prompt
            formatted_prompt = prompt

        # Tokenize with strict max_length enforcement
        # Note: chat templates can expand prompts significantly, so we must truncate
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        ).to(model.device)

        # Generate
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        return response.strip()

    def _compute_linguistic_complexity(self, text: str) -> Dict[str, float]:
        """
        Compute linguistic complexity metrics.

        Args:
            text: Input text

        Returns:
            Dictionary of complexity metrics
        """
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {
                "avg_sentence_length": 0.0,
                "vocabulary_richness": 0.0
            }

        # Average sentence length (words per sentence)
        words_per_sentence = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(words_per_sentence) if words_per_sentence else 0.0

        # Vocabulary richness (unique words / total words)
        all_words = text.lower().split()
        vocabulary_richness = len(set(all_words)) / len(all_words) if all_words else 0.0

        return {
            "avg_sentence_length": float(avg_sentence_length),
            "vocabulary_richness": float(vocabulary_richness)
        }

    def extract_features(
        self,
        model,
        tokenizer,
        prompts: List[str],
        num_probes: int = 100,
        max_new_tokens: int = 64,
        baseline_embeddings: np.ndarray = None,
        seed: int = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract semantic and behavioral features.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompts: List of probe prompts
            num_probes: Number of probes to use
            baseline_embeddings: Optional baseline embeddings for drift comparison

        Returns:
            Dictionary of semantic features
        """
        do_sample = seed is not None
        if do_sample:
            torch.manual_seed(seed)
            logger.info(f"Extracting semantic features with {num_probes} probes "
                        f"(do_sample=True, seed={seed})")
        else:
            logger.info(f"Extracting semantic features with {num_probes} probes (greedy)")

        responses = []
        embeddings_list = []
        complexity_metrics = []

        # Sample prompts
        probe_prompts = prompts[:num_probes]

        for i, prompt in enumerate(probe_prompts):
            try:
                # Generate response
                response = self._generate_response(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                responses.append(response)

                # Compute embedding
                embedding = self.embedding_model.encode(
                    response,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings_list.append(embedding)

                # Compute linguistic complexity
                complexity = self._compute_linguistic_complexity(response)
                complexity_metrics.append(complexity)

                if (i + 1) % 20 == 0:
                    logger.debug(f"Processed {i + 1}/{num_probes} probes")

            except Exception as e:
                logger.warning(f"Failed to process probe {i}: {type(e).__name__}: {str(e)[:100]}")
                continue

        if not embeddings_list:
            logger.error(f"Failed to extract any semantic features out of {len(probe_prompts)} probes")
            raise RuntimeError("Failed to extract any semantic features")

        # Aggregate embeddings
        embeddings_matrix = np.array(embeddings_list)
        mean_embedding = np.mean(embeddings_matrix, axis=0)

        # Compute semantic drift if baseline provided
        semantic_drift_score = 0.0
        if baseline_embeddings is not None:
            semantic_drift_score = compute_cosine_distance(
                mean_embedding,
                baseline_embeddings
            )

        # Aggregate complexity metrics
        avg_sentence_lengths = [m["avg_sentence_length"] for m in complexity_metrics]
        vocabulary_richness_scores = [m["vocabulary_richness"] for m in complexity_metrics]

        features = {
            # Raw per-prompt embedding matrix [num_probes, embedding_dim]
            "semantic_embeddings": embeddings_matrix,

            # Mean embedding vector
            "semantic_embedding_mean": mean_embedding,

            # Embedding variance (semantic consistency)
            "semantic_embedding_std": np.std(embeddings_matrix, axis=0),

            # Semantic drift score (if baseline provided)
            "semantic_drift_score": semantic_drift_score,

            # Linguistic complexity statistics
            "avg_sentence_length_mean": np.mean(avg_sentence_lengths),
            "avg_sentence_length_std": np.std(avg_sentence_lengths),
            "vocabulary_richness_mean": np.mean(vocabulary_richness_scores),
            "vocabulary_richness_std": np.std(vocabulary_richness_scores),

            # Raw responses for analysis
            "responses": responses
        }

        logger.info(f"Extracted semantic features: embedding shape {mean_embedding.shape}")
        return features

    def compare_features(
        self,
        feat1: Dict[str, np.ndarray],
        feat2: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compare semantic features from two model versions.

        Measures how much the behavioral output distributions of V1 and V2 differ:
        - Per-probe cosine similarity between response embeddings
        - Cosine similarity between the cross-model mean embeddings
        - Absolute differences in linguistic complexity statistics

        Args:
            feat1: Output of extract_features() for Model V1.
            feat2: Output of extract_features() for Model V2.

        Returns:
            Dictionary with comparison scalars and per-probe similarity vector.
        """
        emb1 = feat1["semantic_embeddings"]   # [P, D]
        emb2 = feat2["semantic_embeddings"]   # [P, D]

        # Align probe counts
        n = min(len(emb1), len(emb2))
        emb1, emb2 = emb1[:n], emb2[:n]

        # Per-probe cosine similarity
        norms1 = np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8
        norms2 = np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8
        per_probe_cosine = np.sum((emb1 / norms1) * (emb2 / norms2), axis=1)  # [P]

        # Cross-model mean embedding similarity
        mean1 = feat1["semantic_embedding_mean"]
        mean2 = feat2["semantic_embedding_mean"]
        cross_mean_sim = float(np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2) + 1e-8))

        # Linguistic complexity differences
        sl_diff  = abs(feat1["avg_sentence_length_mean"] - feat2["avg_sentence_length_mean"])
        sl_std_diff = abs(feat1["avg_sentence_length_std"] - feat2["avg_sentence_length_std"])
        vr_diff  = abs(feat1["vocabulary_richness_mean"]  - feat2["vocabulary_richness_mean"])
        vr_std_diff = abs(feat1["vocabulary_richness_std"] - feat2["vocabulary_richness_std"])

        return {
            "per_probe_cosine_sim":     per_probe_cosine,
            "mean_cosine_sim":          float(np.mean(per_probe_cosine)),
            "std_cosine_sim":           float(np.std(per_probe_cosine)),
            "min_cosine_sim":           float(np.min(per_probe_cosine)),
            "cross_mean_sim":           cross_mean_sim,
            "sl_diff":                  float(sl_diff),
            "sl_std_diff":              float(sl_std_diff),
            "vr_diff":                  float(vr_diff),
            "vr_std_diff":              float(vr_std_diff),
        }