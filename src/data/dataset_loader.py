"""
Dataset loading and preparation for LIIH probing
"""

import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Optional
import logging
from pathlib import Path

from utils.helpers import format_prompt_for_mmlu

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and prepares probe datasets for LIIH signature generation.
    Supports MMLU-Pro and TruthfulQA datasets.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize dataset loader.

        Args:
            cache_dir: Directory to cache processed datasets
        """
        self.cache_dir = cache_dir
        self.loaded_datasets: Dict[str, pd.DataFrame] = {}

    def load_mmlu_pro(
        self,
        num_samples: int = 100,
        use_cache: bool = True,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load MMLU-Pro dataset for complex reasoning probes.

        Args:
            num_samples: Number of samples to load
            use_cache: Whether to use cached version if available
            random_state: Random seed for sampling

        Returns:
            DataFrame with 'prompt' column
        """
        cache_path = None
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"mmlu_pro_{num_samples}.csv"
            if cache_path.exists():
                logger.info(f"Loading cached MMLU-Pro from {cache_path}")
                df = pd.read_csv(cache_path)
                self.loaded_datasets["mmlu_pro"] = df
                return df

        logger.info("Loading MMLU-Pro dataset from HuggingFace")

        try:
            # Load dataset
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            df = ds.to_pandas()

            # Sample and format
            sampled_df = df.sample(n=min(num_samples, len(df)), random_state=random_state)
            sampled_df = sampled_df.copy()  # Avoid SettingWithCopyWarning
            sampled_df['prompt'] = sampled_df.apply(format_prompt_for_mmlu, axis=1)

            # Cache if requested
            if use_cache and cache_path:
                sampled_df.to_csv(cache_path, index=False, encoding='utf-8-sig')
                logger.info(f"Cached MMLU-Pro to {cache_path}")

            self.loaded_datasets["mmlu_pro"] = sampled_df
            logger.info(f"Loaded {len(sampled_df)} MMLU-Pro samples")
            return sampled_df

        except Exception as e:
            logger.error(f"Failed to load MMLU-Pro: {e}")
            raise

    def load_truthfulqa(
        self,
        num_samples: int = 50,
        use_cache: bool = True,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load TruthfulQA dataset for adversarial/factuality probes.

        Args:
            num_samples: Number of samples to load
            use_cache: Whether to use cached version if available
            random_state: Random seed for sampling

        Returns:
            DataFrame with 'prompt' column
        """
        cache_path = None
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"truthfulqa_{num_samples}.csv"
            if cache_path.exists():
                logger.info(f"Loading cached TruthfulQA from {cache_path}")
                df = pd.read_csv(cache_path)
                self.loaded_datasets["truthfulqa"] = df
                return df

        logger.info("Loading TruthfulQA dataset from HuggingFace")

        try:
            # Load dataset
            ds = load_dataset("truthful_qa", "generation", split="validation")
            df = ds.to_pandas()

            # Sample and format
            sampled_df = df.sample(n=min(num_samples, len(df)), random_state=random_state)
            sampled_df = sampled_df.copy()

            # Use 'question' field as prompt
            sampled_df['prompt'] = sampled_df['question']

            # Cache if requested
            if use_cache and cache_path:
                sampled_df.to_csv(cache_path, index=False, encoding='utf-8-sig')
                logger.info(f"Cached TruthfulQA to {cache_path}")

            self.loaded_datasets["truthfulqa"] = sampled_df
            logger.info(f"Loaded {len(sampled_df)} TruthfulQA samples")
            return sampled_df

        except Exception as e:
            logger.error(f"Failed to load TruthfulQA: {e}")
            raise

    def load_custom_probes(self, prompts: List[str]) -> pd.DataFrame:
        """
        Create a dataset from a custom list of prompts.

        Args:
            prompts: List of prompt strings

        Returns:
            DataFrame with 'prompt' column
        """
        df = pd.DataFrame({"prompt": prompts})
        self.loaded_datasets["custom"] = df
        logger.info(f"Loaded {len(df)} custom prompts")
        return df

    def get_combined_probes(
        self,
        datasets: List[str] = ["mmlu_pro", "truthfulqa"],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load and combine multiple probe datasets.

        Args:
            datasets: List of dataset names to combine
            **kwargs: Arguments passed to individual loaders

        Returns:
            Combined DataFrame with 'prompt' and 'source' columns
        """
        combined_dfs = []

        for ds_name in datasets:
            if ds_name == "mmlu_pro":
                df = self.load_mmlu_pro(**kwargs)
                df['source'] = 'mmlu_pro'
                combined_dfs.append(df[['prompt', 'source']])
            elif ds_name == "truthfulqa":
                df = self.load_truthfulqa(**kwargs)
                df['source'] = 'truthfulqa'
                combined_dfs.append(df[['prompt', 'source']])

        if not combined_dfs:
            raise ValueError("No datasets loaded")

        combined_df = pd.concat(combined_dfs, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} probes from {len(datasets)} datasets")
        return combined_df

    def get_prompts_list(self, dataset_name: str) -> List[str]:
        """
        Get list of prompts from a loaded dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of prompt strings
        """
        if dataset_name not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        return self.loaded_datasets[dataset_name]['prompt'].tolist()
