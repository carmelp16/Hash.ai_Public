"""
Semantic Perturbation for ZeroPrint Fingerprinting
Based on carmel notebooks and LIIH Framework
"""

import torch
from transformers import pipeline
from typing import Optional


class ProductionPerturber:
    """
    Performs semantic-preserving perturbations on text prompts
    for zeroth-order gradient estimation (ZeroPrint methodology).

    Uses a masked language model to find contextual synonyms,
    enabling the approximation of the Jacobian matrix without
    requiring white-box access to model gradients.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", device: Optional[str] = None):
        """
        Initialize the perturber with a masked language model.

        Args:
            model_name: HuggingFace model for fill-mask task
            device: Device to run on ("cuda", "cpu", or None for auto-detect)
        """
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            device = 0
        elif device == "cpu":
            device = -1

        self.mask_filler = pipeline(
            "fill-mask",
            model=model_name,
            device=device
        )

    def get_semantic_perturbation(self, prompt: str, target_position: str = "middle") -> str:
        """
        Generate a semantically-similar variant of the prompt by substituting
        a target word with a contextual synonym.

        Args:
            prompt: Original text prompt
            target_position: Which word to perturb ("middle", "first_noun", "random")

        Returns:
            Perturbed prompt with one word substituted
        """
        words = prompt.split()

        # Fallback for very short prompts
        if len(words) < 3:
            return prompt + " "

        # Token-aware truncation: DistilBERT has a 512-token hard limit.
        # Word count alone is unreliable because subword tokenization can
        # expand a single word to multiple tokens. Truncate on actual tokens,
        # reserving 3 slots for [CLS], [SEP], and the [MASK] token itself.
        max_tokens = 509
        token_ids = self.mask_filler.tokenizer.encode(
            " ".join(words), add_special_tokens=False
        )
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
            truncated_text = self.mask_filler.tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            words = truncated_text.split()

        # Select target word index
        if target_position == "middle":
            target_idx = len(words) // 2
        elif target_position == "first_noun":
            # Simple heuristic: skip first word if it's "The", "A", etc.
            target_idx = 1 if words[0].lower() in ["the", "a", "an"] else 0
        else:  # random
            import random
            target_idx = random.randint(1, len(words) - 2)

        original_word = words[target_idx]

        # Create masked prompt
        masked_words = words.copy()
        masked_words[target_idx] = self.mask_filler.tokenizer.mask_token
        masked_prompt = " ".join(masked_words)

        try:
            suggestions = self.mask_filler(masked_prompt, top_k=5)

            # Pick best synonym that differs from original
            best_synonym = original_word
            for suggestion in suggestions:
                candidate = suggestion['token_str'].strip().lower()

                # Avoid identical words, punctuation, and very short tokens
                if (candidate != original_word.lower() and
                    len(candidate) > 1 and
                    candidate.isalpha()):
                    best_synonym = candidate
                    break

            # Reconstruct perturbed prompt
            words[target_idx] = best_synonym
            return " ".join(words)

        except Exception as e:
            # Fallback: return original with slight modification
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Perturbation failed for '{prompt[:50]}...': {type(e).__name__}: {str(e)}")
            return prompt + " "

    def get_multiple_perturbations(self, prompt: str, num_variants: int = 5) -> list[str]:
        """
        Generate multiple semantic perturbations of the same prompt.

        Args:
            prompt: Original text prompt
            num_variants: Number of variants to generate

        Returns:
            List of perturbed prompts
        """
        variants = []
        positions = ["middle"] * num_variants  # Can vary strategy if needed

        for pos in positions:
            variant = self.get_semantic_perturbation(prompt, target_position=pos)
            if variant not in variants:  # Avoid duplicates
                variants.append(variant)

        return variants[:num_variants]
