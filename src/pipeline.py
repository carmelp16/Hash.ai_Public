"""
LLM Hash Verification Pipeline
Verifies whether a "new version" of a trusted Black-Box LLM has been tampered with.

Each training sample is a (Model_V1, Model_V2) pair.
The pipeline extracts LIIH signatures for both models, computes comparison features,
then trains a binary classifier:
  Label 0 = V2 is legitimate (same weights as V1)
  Label 1 = V2 has been tampered (fine-tuned, backdoored, quantized, etc.)
"""

import gc
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('llm_hash_verifier.log')
    ]
)
logger = logging.getLogger(__name__)

from config import (
    MODEL_PAIRS, PROBE_DATASETS,
    JACOBIAN_CONFIG, SEMANTIC_CONFIG, TEMPORAL_CONFIG, LLMMAP_CONFIG,
    CLASSIFIER_CONFIGS, PIPELINE_CONFIG,
    BACKDOOR_CONFIG, MODELS_TO_BACKDOOR,
    RANDOM_SEED, RESULTS_DIR, MODELS_DIR
)
from data import ModelLoader, DatasetLoader, BackdoorInjector
from features import LIIHFeatureBuilder
from classifier import ClassifierTrainer, ClassifierEvaluator
from utils import set_random_seed


class LLMHashVerificationPipeline:
    """
    End-to-end pipeline for LLM integrity verification using LIIH comparison signatures.

    For each (V1, V2) model pair:
      1. Load V1 → extract LIIH signature → unload V1
      2. Load V2 → extract LIIH signature → unload V2
      3. Compute comparison feature vector from the two signatures
      4. Assign pair label (0=legitimate, 1=tampered)

    The feature matrix (pairs × comparison_dims) is then fed to a binary classifier.
    """

    def __init__(self):
        logger.info("=" * 80)
        logger.info("LLM HASH VERIFICATION PIPELINE")
        logger.info("Based on LIIH (Layered Integrity Invariant Hash) Framework")
        logger.info("=" * 80)

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("Authenticated with HuggingFace Hub")
        else:
            logger.warning("HF_TOKEN not set — gated models will fail to download")

        set_random_seed(RANDOM_SEED)

        self.model_loader   = ModelLoader()
        self.dataset_loader = DatasetLoader(cache_dir=PIPELINE_CONFIG["cache_dir"])
        self.feature_builder = LIIHFeatureBuilder(
            JACOBIAN_CONFIG, SEMANTIC_CONFIG, TEMPORAL_CONFIG, LLMMAP_CONFIG
        )
        self.evaluator = ClassifierEvaluator(RESULTS_DIR)

        self.pairs_cache_path = PIPELINE_CONFIG["cache_dir"] / "pair_features.pkl"

    # -------------------------------------------------------------------------
    # Step 0 — Backdoor injection (creates and caches backdoored model variants)
    # -------------------------------------------------------------------------

    def create_backdoored_pairs(self) -> list:
        """
        For each model in MODELS_TO_BACKDOOR:
          1. Check whether a cached backdoored variant already exists on disk.
          2. If not: load the clean model, run BackdoorInjector.inject(), save.
          3. Build a pair entry  (v1=clean_hf_id, v2=local_path, label=1).

        Returns a list of pair dicts in the same format as MODEL_PAIRS that can
        be appended before feature extraction.  Only successfully injected models
        are included; failures are logged and skipped.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 0: CREATING BACKDOORED MODEL VARIANTS")
        logger.info("=" * 80)

        injector = BackdoorInjector(
            trigger=BACKDOOR_CONFIG["trigger"],
            target_output=BACKDOOR_CONFIG["target_output"],
            num_steps=BACKDOOR_CONFIG["num_steps"],
            lr=BACKDOOR_CONFIG["lr"],
            gradient_accumulation_steps=BACKDOOR_CONFIG["gradient_accumulation_steps"],
            poison_fraction=BACKDOOR_CONFIG["poison_fraction"],
        )

        save_root = BACKDOOR_CONFIG["save_dir"]
        backdoor_pairs = []

        for clean_cfg in MODELS_TO_BACKDOOR:
            model_name  = clean_cfg["name"]
            backdoor_name = f"{model_name}-backdoored"
            save_path     = save_root / backdoor_name

            logger.info(f"\n  Model: {model_name}")

            try:
                # --- Load or create the backdoored variant ---
                if save_path.exists():
                    logger.info(f"  Cached backdoored model found at {save_path} — skipping injection")
                else:
                    logger.info(f"  No cache found — loading clean model and injecting backdoor")
                    use_offline = PIPELINE_CONFIG.get("use_offline_mode", False)
                    model, tokenizer = self.model_loader.load_model(
                        clean_cfg, local_files_only=use_offline
                    )

                    # Inject backdoor in-place
                    model = injector.inject(model, tokenizer, model_name=model_name)

                    # Optional ASR verification
                    if BACKDOOR_CONFIG.get("verify_after_injection", True):
                        asr = injector.verify_backdoor(model, tokenizer, num_checks=5)
                        logger.info(f"  Attack Success Rate (ASR): {asr:.1%}")

                    # Save and unload
                    injector.save(model, tokenizer, save_path)
                    self.model_loader.unload_model(model_name)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                # Build pair entry (v2 references the local save path)
                backdoor_pairs.append({
                    "v1": clean_cfg,
                    "v2": {
                        "name":              backdoor_name,
                        "hf_name":           str(save_path),   # local path works with from_pretrained
                        "trust_remote_code": clean_cfg.get("trust_remote_code", False),
                    },
                    "label":             1,
                    "modification_type": "backdoor",
                })
                logger.info(f"  Backdoor pair registered: {model_name} → {backdoor_name}")

            except Exception as e:
                logger.error(f"  Failed to create backdoor for {model_name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.info(f"\nBackdoor pairs created: {len(backdoor_pairs)}/{len(MODELS_TO_BACKDOOR)}")
        return backdoor_pairs

    # -------------------------------------------------------------------------
    # Step 1 — Probe datasets
    # -------------------------------------------------------------------------

    def load_probe_datasets(self):
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOADING PROBE DATASETS")
        logger.info("=" * 80)

        all_prompts = []

        if "mmlu_pro" in PROBE_DATASETS:
            cfg = PROBE_DATASETS["mmlu_pro"]
            df  = self.dataset_loader.load_mmlu_pro(
                num_samples=cfg["num_samples"],
                use_cache=PIPELINE_CONFIG["cache_features"]
            )
            all_prompts.extend(df['prompt'].tolist())

        if "truthfulqa" in PROBE_DATASETS:
            cfg = PROBE_DATASETS["truthfulqa"]
            df  = self.dataset_loader.load_truthfulqa(
                num_samples=cfg["num_samples"],
                use_cache=PIPELINE_CONFIG["cache_features"]
            )
            all_prompts.extend(df['prompt'].tolist())

        logger.info(f"Total probe prompts: {len(all_prompts)}")
        return all_prompts

    # -------------------------------------------------------------------------
    # Step 2 — Extract LIIH signatures for all pairs (one model at a time)
    # -------------------------------------------------------------------------

    def _extract_single_signature(self, model_config: dict, prompts: list) -> dict:
        """Load one model, extract its LIIH signature, then unload it.

        If `model_config` contains an `inference_seed` key, generation is
        stochastic (do_sample=True, temperature=0.8) with that seed.
        Otherwise, greedy decoding is used (do_sample=False).
        """
        use_offline = PIPELINE_CONFIG.get("use_offline_mode", False)
        model, tokenizer = self.model_loader.load_model(
            model_config, local_files_only=use_offline
        )
        seed = model_config.get("inference_seed")
        sig = self.feature_builder.extract_all_features(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            model_name=model_config["name"],
            seed=seed,
        )
        self.model_loader.unload_model(model_config["name"])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return sig

    def extract_pair_features(self, prompts: list, pairs: list = None, use_cache: bool = True) -> dict:
        """
        Iterate over MODEL_PAIRS and build comparison feature vectors.

        Returns
        -------
        pairs_features : dict
            Maps pair_id (str) → {
                "sig1":             raw LIIH signature for V1,
                "sig2":             raw LIIH signature for V2,
                "comparison_vector": np.ndarray,
                "label":            int (0 or 1),
                "modification_type": str,
                "feature_names":    list[str],
            }
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: EXTRACTING LIIH COMPARISON FEATURES FOR ALL PAIRS")
        logger.info("=" * 80)

        if use_cache and self.pairs_cache_path.exists():
            logger.info(f"Loading cached pair features from {self.pairs_cache_path}")
            return joblib.load(self.pairs_cache_path)

        if pairs is None:
            pairs = MODEL_PAIRS

        pairs_features = {}
        feature_names  = self.feature_builder.get_feature_names()

        for i, pair in enumerate(pairs):
            v1_cfg  = pair["v1"]
            v2_cfg  = pair["v2"]
            label   = pair["label"]
            mod_type = pair["modification_type"]
            pair_id  = f"{v1_cfg['name']}__vs__{v2_cfg['name']}"

            logger.info(f"\n{'='*60}")
            logger.info(f"Pair {i+1}/{len(pairs)}: {pair_id}  [label={label}, type={mod_type}]")
            logger.info(f"{'='*60}")

            try:
                logger.info(f"  Loading V1: {v1_cfg['name']} "
                            f"(seed={v1_cfg.get('inference_seed', 'greedy')})")
                sig1 = self._extract_single_signature(v1_cfg, prompts)

                logger.info(f"  Loading V2: {v2_cfg['name']} "
                            f"(seed={v2_cfg.get('inference_seed', 'greedy')})")
                sig2 = self._extract_single_signature(v2_cfg, prompts)

                comparison_vector = self.feature_builder.build_comparison_vector(sig1, sig2)

                pairs_features[pair_id] = {
                    "sig1":              sig1,
                    "sig2":              sig2,
                    "comparison_vector": comparison_vector,
                    "label":             label,
                    "modification_type": mod_type,
                    "feature_names":     feature_names,
                }

                logger.info(f"  Comparison vector: {len(comparison_vector)} features")

            except Exception as e:
                logger.error(f"Failed to process pair {pair_id}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        if not pairs_features:
            raise RuntimeError("Failed to extract features from any model pair")

        logger.info(f"\nExtracted features for {len(pairs_features)} pairs")

        if PIPELINE_CONFIG["cache_features"]:
            joblib.dump(pairs_features, self.pairs_cache_path)
            logger.info(f"Pair features cached to {self.pairs_cache_path}")

        return pairs_features

    # -------------------------------------------------------------------------
    # Shared train/test split (stratified by modification_type × label)
    # -------------------------------------------------------------------------

    def _make_split(self, pairs_features: dict):
        """
        Build a train/test split stratified by modification_type × label so that
        every modification type is represented equally in both sets.

        Fallback chain:
          1. Stratify by modification_type × label  (ideal)
          2. Stratify by label only                 (if any stratum has < 2 samples)
          3. Random split                           (last resort)

        After splitting, logs every pair name grouped by modification type for
        both the train and test sets.

        Returns
        -------
        pair_ids  : list[str]
        train_idx : np.ndarray of int
        test_idx  : np.ndarray of int
        """
        from collections import Counter, defaultdict

        pair_ids  = list(pairs_features.keys())
        y         = np.array([pairs_features[pid]["label"]             for pid in pair_ids])
        mod_types = [pairs_features[pid]["modification_type"] for pid in pair_ids]

        # Combined stratification key
        strat_keys = [f"{y[i]}__{mod_types[i]}" for i in range(len(pair_ids))]

        test_size    = CLASSIFIER_CONFIGS[0]["test_size"]
        random_state = CLASSIFIER_CONFIGS[0]["random_state"]
        indices      = np.arange(len(pair_ids))

        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state,
                stratify=strat_keys
            )
            logger.info("Split stratified by modification_type × label")
        except ValueError:
            logger.warning(
                "Cannot stratify by modification_type × label "
                f"(counts: {dict(Counter(strat_keys))}) — "
                "falling back to label-only stratification"
            )
            try:
                train_idx, test_idx = train_test_split(
                    indices, test_size=test_size, random_state=random_state,
                    stratify=y
                )
                logger.info("Split stratified by label only")
            except ValueError:
                logger.warning("Cannot stratify by label — using random split")
                train_idx, test_idx = train_test_split(
                    indices, test_size=test_size, random_state=random_state
                )

        # ---- Log which pairs landed in each set ----
        def _log_set(set_name, idx_arr):
            by_type = defaultdict(list)
            for i in idx_arr:
                pid   = pair_ids[i]
                mtype = pairs_features[pid]["modification_type"]
                label = pairs_features[pid]["label"]
                by_type[(mtype, label)].append(pid)

            label_str_map = {0: "legitimate", 1: "tampered"}
            logger.info(f"\n  {set_name} SET ({len(idx_arr)} pairs):")
            for (mtype, lbl), pids in sorted(by_type.items()):
                logger.info(f"    [{mtype} / {label_str_map[lbl]}]  ({len(pids)} pairs)")
                for pid in pids:
                    v1, v2 = pid.split("__vs__")
                    logger.info(f"      {v1}  →  {v2}")

        logger.info("\n" + "=" * 80)
        logger.info("TRAIN / TEST SPLIT")
        logger.info("=" * 80)
        _log_set("TRAIN", train_idx)
        _log_set("TEST",  test_idx)

        return pair_ids, train_idx, test_idx

    # -------------------------------------------------------------------------
    # Step 3 — Train classifiers
    # -------------------------------------------------------------------------

    def train_classifier(self, pairs_features: dict):
        """Train all configured classifiers on a shared train/test split."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: TRAINING CLASSIFIERS")
        logger.info("=" * 80)

        pair_ids, train_idx, test_idx = self._make_split(pairs_features)

        X = np.array([pairs_features[pid]["comparison_vector"] for pid in pair_ids])
        y = np.array([pairs_features[pid]["label"]             for pid in pair_ids])
        feature_names = pairs_features[pair_ids[0]].get("feature_names")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"\nDataset: {X.shape[0]} pairs × {X.shape[1]} features")
        logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")
        logger.info(f"Class distribution — legitimate: {(y==0).sum()}, tampered: {(y==1).sum()}")

        results = []
        for clf_config in CLASSIFIER_CONFIGS:
            algo = clf_config["algorithm"]
            logger.info(f"\n{'='*60}\nTraining: {algo.upper()}\n{'='*60}")

            trainer = ClassifierTrainer(clf_config)
            train_metrics = trainer.train(X_train, y_train, perform_cv=True)
            test_metrics  = trainer.evaluate(X_test, y_test)
            importance_df = trainer.get_feature_importance(feature_names)

            model_path = MODELS_DIR / f"classifier_{algo}.pkl"
            trainer.save_model(model_path)

            results.append((algo, train_metrics, test_metrics, importance_df))
            logger.info(f"{algo} — test accuracy: {test_metrics['test_accuracy']:.4f}")

        return results, (X_test, y_test)

    # -------------------------------------------------------------------------
    # Step 3b — Component ablation study
    # -------------------------------------------------------------------------

    def run_ablation_study(self, pairs_features: dict):
        """
        Leave-one-component-out ablation using cached pair signatures.
        No model re-inference required.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3b: COMPONENT-LEVEL ABLATION STUDY")
        logger.info("=" * 80)

        rf_config = next(c for c in CLASSIFIER_CONFIGS if c["algorithm"] == "random_forest")

        pair_ids, train_idx, test_idx = self._make_split(pairs_features)
        y = np.array([pairs_features[pid]["label"] for pid in pair_ids])
        y_train, y_test = y[train_idx], y[test_idx]

        component_labels = {
            "I":  "Drop Component I  (Jacobian)",
            "BS": "Drop Component BS (Semantic)",
            "T":  "Drop Component T  (Temporal)",
            "L":  "Drop Component L  (LLMmap)",
        }

        ablation_results = []
        for drop, label_str in component_labels.items():
            logger.info(f"\nAblation: {label_str}")

            ablated_vecs = self.feature_builder.build_ablated_composite_vectors(
                {pid: {"sig1": pairs_features[pid]["sig1"], "sig2": pairs_features[pid]["sig2"]}
                 for pid in pair_ids},
                drop_component=drop
            )
            X_abl = np.array([ablated_vecs[pid] for pid in pair_ids])

            trainer = ClassifierTrainer(rf_config)
            train_metrics = trainer.train(X_abl[train_idx], y_train, perform_cv=True)
            test_metrics  = trainer.evaluate(X_abl[test_idx], y_test)

            ablation_results.append({
                "drop":         drop,
                "label":        label_str,
                "cv_mean":      train_metrics["cv_mean"],
                "cv_std":       train_metrics["cv_std"],
                "test_accuracy": test_metrics["test_accuracy"],
            })
            logger.info(
                f"  CV: {train_metrics['cv_mean']:.4f} ± {train_metrics['cv_std']:.4f} "
                f"| Test: {test_metrics['test_accuracy']:.4f}"
            )

        logger.info("\nAblation Summary:")
        logger.info(f"{'Dropped Component':<38} {'CV Accuracy':>15} {'Test Accuracy':>15}")
        logger.info("-" * 70)
        for r in ablation_results:
            cv_str = f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}" if r["cv_mean"] is not None else "N/A"
            logger.info(f"{r['label']:<38} {cv_str:>15} {r['test_accuracy']:>15.4f}")

        return ablation_results

    # -------------------------------------------------------------------------
    # Step 4 — Report
    # -------------------------------------------------------------------------

    def generate_report(self, classifier_name, train_metrics, test_metrics, importance_df):
        self.evaluator.create_full_report(
            train_metrics, test_metrics, importance_df,
            classifier_name=classifier_name
        )

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(self):
        try:
            # Step 0 — create backdoored model variants and add them to the pairs list
            backdoor_pairs = self.create_backdoored_pairs()
            all_pairs = MODEL_PAIRS + backdoor_pairs

            n_legit    = sum(1 for p in all_pairs if p["label"] == 0)
            n_tampered = sum(1 for p in all_pairs if p["label"] == 1)
            logger.info(f"\nTotal model pairs to process: {len(all_pairs)}")
            logger.info(f"  Legitimate (label=0): {n_legit}")
            logger.info(f"  Tampered   (label=1): {n_tampered}")
            logger.info(f"  Of which backdoored:  {len(backdoor_pairs)}")

            # Step 1
            prompts = self.load_probe_datasets()

            # Step 2
            pairs_features = self.extract_pair_features(
                prompts, pairs=all_pairs, use_cache=PIPELINE_CONFIG["cache_features"]
            )

            # Step 3
            results, test_data = self.train_classifier(pairs_features)

            # Step 3b
            ablation_results = self.run_ablation_study(pairs_features)

            # Step 4
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: GENERATING EVALUATION REPORTS")
            logger.info("=" * 80)
            for algo, train_metrics, test_metrics, importance_df in results:
                self.generate_report(algo, train_metrics, test_metrics, importance_df)

            # Final summary
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"\n{'Classifier':<25} {'Test Accuracy':>15}")
            logger.info("-" * 42)
            for algo, _, test_metrics, _ in results:
                logger.info(f"{algo:<25} {test_metrics['test_accuracy']:>15.4f}")
            logger.info(f"\nReports saved to: {RESULTS_DIR}")

        except Exception as e:
            logger.error(f"\nPipeline failed: {e}", exc_info=True)
            raise
        finally:
            logger.info("\nCleaning up models from memory...")
            self.model_loader.unload_all_models()


def main():
    pipeline = LLMHashVerificationPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
