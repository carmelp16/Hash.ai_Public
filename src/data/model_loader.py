"""
Model loading and management for benign and modified LLMs
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles loading of benign and modified LLMs from HuggingFace.
    Manages tokenizers and models for the binary classifier pipeline.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            device: Device to load models on ("cuda", "mps", "cpu", or None for auto)
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"ModelLoader using device: {self.device}")

        self.loaded_models: Dict[str, dict] = {}

    def load_model(
        self,
        model_config: dict,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        local_files_only: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a single model and tokenizer from HuggingFace.

        Args:
            model_config: Model configuration dictionary with keys:
                - name: Short name for the model
                - hf_name: HuggingFace model identifier
                - trust_remote_code: Whether to trust remote code
            load_in_8bit: Load model in 8-bit precision (quantization)
            load_in_4bit: Load model in 4-bit precision (quantization)
            local_files_only: Use only locally cached files (offline mode)

        Returns:
            Tuple of (model, tokenizer)
        """
        name = model_config["name"]
        hf_name = model_config["hf_name"]
        trust_remote_code = model_config.get("trust_remote_code", False)

        logger.info(f"Loading model: {name} ({hf_name}){' [OFFLINE MODE]' if local_files_only else ''}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                hf_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only
            )

            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load config first to fix pad_token_id if needed
            config = AutoConfig.from_pretrained(
                hf_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only
            )

            # Fix pad_token_id for models that don't have it (e.g., Phi models)
            if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
                if hasattr(config, 'eos_token_id') and config.eos_token_id is not None:
                    config.pad_token_id = config.eos_token_id
                elif tokenizer.eos_token_id is not None:
                    config.pad_token_id = tokenizer.eos_token_id
                else:
                    # Fallback to 0 if nothing else is available
                    config.pad_token_id = 0

            # Fix rope_scaling for Phi-3, Llama-3.2, and other models
            if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
                rs = config.rope_scaling
                # 1. Normalize any single-element list values to scalars.
                #    Some Llama 3.2 checkpoints store e.g. "factor": [32.0] instead of 32.0,
                #    which causes "'<' not supported between instances of 'list' and 'int'"
                #    in the transformers rope_scaling validator.
                for key, val in list(rs.items()):
                    if isinstance(val, list) and len(val) == 1:
                        rs[key] = val[0]
                # 2. Ensure the 'type' key is present.
                #    Llama 3.2 uses 'rope_type' instead of 'type'; alias it.
                if 'type' not in rs:
                    rs['type'] = rs.get('rope_type', 'default')
            elif hasattr(config, 'rope_scaling') and config.rope_scaling is None:
                config.rope_scaling = {'type': 'default', 'factor': 1.0}

            # Load model
            # Use "auto" for CUDA (handles multi-GPU sharding);
            # use explicit mapping for MPS/CPU to avoid accelerate issues.
            if self.device == "cuda":
                device_map = "auto"
            else:
                device_map = {"": self.device}

            model_kwargs = {
                "config": config,
                "trust_remote_code": trust_remote_code,
                "device_map": device_map,
                "local_files_only": local_files_only
            }

            # Quantization options — explicit params take precedence over config key
            quant = model_config.get("quantization")
            if load_in_8bit or quant == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit or quant == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            else:
                model_kwargs["torch_dtype"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(
                hf_name,
                **model_kwargs
            )

            model.eval()  # Set to evaluation mode

            # Cache loaded model
            self.loaded_models[name] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": model_config,
                "quantization": model_config.get("quantization") or ("int8" if load_in_8bit else "int4" if load_in_4bit else "full")
            }

            logger.info(f"Successfully loaded {name}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            raise

    def load_benign_models(self, model_configs: List[dict]) -> Dict[str, dict]:
        """
        Load all benign (standard) models.

        Args:
            model_configs: List of model configuration dictionaries

        Returns:
            Dictionary mapping model names to {model, tokenizer, config}
        """
        benign_models = {}

        for config in model_configs:
            try:
                model, tokenizer = self.load_model(config)
                benign_models[config["name"]] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": config,
                    "label": 0  # Benign label
                }
            except Exception as e:
                logger.warning(f"Skipping model {config['name']} due to error: {e}")
                continue

        if not benign_models:
            raise RuntimeError("Failed to load any benign models")

        logger.info(f"Loaded {len(benign_models)} benign models")
        return benign_models

    def load_modified_models(self, model_configs: List[dict]) -> Dict[str, dict]:
        """
        Load all modified (potentially malicious) models.

        Args:
            model_configs: List of model configuration dictionaries

        Returns:
            Dictionary mapping model names to {model, tokenizer, config}
        """
        modified_models = {}

        for config in model_configs:
            try:
                # Check if we should simulate quantization
                modification_type = config.get("modification_type", "substitution")

                if modification_type == "quantization":
                    model, tokenizer = self.load_model(config, load_in_8bit=True)
                else:  # substitution or other
                    model, tokenizer = self.load_model(config)

                modified_models[config["name"]] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": config,
                    "label": 1  # Modified/malicious label
                }
            except Exception as e:
                logger.warning(f"Skipping model {config['name']} due to error: {e}")
                continue

        if not modified_models:
            raise RuntimeError("Failed to load any modified models")

        logger.info(f"Loaded {len(modified_models)} modified models")
        return modified_models

    def get_all_models(self) -> Dict[str, dict]:
        """
        Get all loaded models (both benign and modified).

        Returns:
            Dictionary of all loaded models
        """
        return self.loaded_models

    def unload_model(self, model_name: str):
        """
        Unload a specific model to free memory.

        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.loaded_models:
            entry = self.loaded_models.pop(model_name)
            entry.pop("model", None)
            entry.pop("tokenizer", None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info(f"Unloaded model: {model_name}")

    def unload_all_models(self):
        """
        Unload all models to free memory.
        """
        for name in list(self.loaded_models.keys()):
            self.unload_model(name)
        self.loaded_models.clear()
        logger.info("All models unloaded")
