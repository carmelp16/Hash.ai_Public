"""
Binary Classifier Training Module
Trains Random Forest/XGBoost classifier on LIIH features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
import joblib
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """
    Trains binary classifier to verify LLM integrity based on LIIH comparison vectors.

    Each training sample is a (V1, V2) model pair represented by its comparison
    feature vector.  Label 0 = legitimate (V2 is V1), label 1 = tampered.
    """

    def __init__(self, config: dict):
        """
        Initialize classifier trainer.

        Args:
            config: Classifier configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler() if config.get("scale_features", True) else None
        self.feature_selector = None
        self.model = None

    def _create_classifier(self):
        """
        Create classifier based on configuration.

        Returns:
            Classifier object
        """
        algorithm = self.config.get("algorithm", "random_forest")

        if algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.get("n_estimators", 100),
                max_depth=self.config.get("max_depth", 10),
                random_state=self.config.get("random_state", 42),
                n_jobs=-1
            )
        elif algorithm == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=self.config.get("n_estimators", 100),
                    max_depth=self.config.get("max_depth", 10),
                    random_state=self.config.get("random_state", 42),
                    n_jobs=-1
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                return RandomForestClassifier(
                    n_estimators=self.config.get("n_estimators", 100),
                    max_depth=self.config.get("max_depth", 10),
                    random_state=self.config.get("random_state", 42),
                    n_jobs=-1
                )
        elif algorithm == "svm":
            from sklearn.svm import SVC
            return SVC(
                kernel=self.config.get("kernel", "rbf"),
                C=self.config.get("C", 1.0),
                probability=True,  # required for predict_proba
                random_state=self.config.get("random_state", 42)
            )
        elif algorithm == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=self.config.get("C", 1.0),
                max_iter=self.config.get("max_iter", 1000),
                random_state=self.config.get("random_state", 42),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def prepare_dataset(
        self,
        features_dict: Dict[str, Dict]
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare training dataset from extracted features.

        Args:
            features_dict: Dictionary mapping model names to feature dictionaries

        Returns:
            Tuple of (X, y, model_names)
        """
        X_list = []
        y_list = []
        names_list = []

        for model_name, features in features_dict.items():
            # Get composite vector and label
            X_list.append(features["composite_vector"])
            y_list.append(features["label"])
            names_list.append(model_name)

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Prepared dataset: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")

        return X, y, names_list

    def _create_feature_selector(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Create feature selector based on configuration.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Feature selector object
        """
        if not self.config.get("use_feature_selection", False):
            return None

        method = self.config.get("feature_selection_method", "rf_importance")
        n_features = self.config.get("n_features_to_select", 100)

        # Ensure n_features doesn't exceed available features
        n_features = min(n_features, X_train.shape[1])

        logger.info(f"Creating feature selector: {method}, selecting {n_features} features")

        variance_ratio = self.config.get("pca_variance_ratio", 0.95)
        random_state = self.config.get("random_state", 42)

        if method == "rf_importance":
            rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state, n_jobs=-1)
            selector = RFE(estimator=rf, n_features_to_select=n_features, step=10)

        elif method == "mutual_info":
            # The comparison feature vector is ~100-dimensional — PCA is unnecessary
            # and unstable at this scale.  Use SelectKBest(mutual_info) directly.
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            logger.info(f"Using SelectKBest(mutual_info, k={n_features})")

        elif method == "pca":
            selector = PCA(n_components=variance_ratio, random_state=random_state)
            logger.info(f"Using PCA with {variance_ratio*100}% variance ratio")

        else:
            logger.warning(f"Unknown feature selection method: {method}, disabling feature selection")
            return None

        # Fit the selector (Pipeline.fit handles both stages in sequence)
        selector.fit(X_train, y_train)

        if hasattr(selector, 'n_components_'):
            logger.info(f"PCA selected {selector.n_components_} components")
        elif hasattr(selector, 'n_features_in_'):
            logger.info(f"Feature selector fitted on {selector.n_features_in_} input features")

        return selector

    def apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """
        Apply feature selection/reduction to features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        if self.feature_selector is None:
            return X

        return self.feature_selector.transform(X)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        perform_cv: bool = True
    ) -> Dict[str, float]:
        """
        Train the binary classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            perform_cv: Whether to perform cross-validation

        Returns:
            Dictionary with training metrics
        """
        logger.info("Training binary classifier")

        # Scale features if configured
        if self.scaler is not None:
            logger.info("Scaling features")
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train

        # Apply feature selection if configured
        self.feature_selector = self._create_feature_selector(X_train_scaled, y_train)
        if self.feature_selector is not None:
            logger.info(f"Original feature shape: {X_train_scaled.shape}")
            X_train_scaled = self.apply_feature_selection(X_train_scaled)
            logger.info(f"Selected feature shape: {X_train_scaled.shape}")

        # Create and train model
        self.model = self._create_classifier()
        self.model.fit(X_train_scaled, y_train)

        logger.info("Training completed")

        # Cross-validation if requested
        cv_scores = None
        if perform_cv:
            cv_folds = self.config.get("cv_folds", 5)

            # Check if we have enough samples for cross-validation
            min_samples_required = cv_folds
            if len(X_train_scaled) < min_samples_required:
                logger.warning(
                    f"Not enough samples for {cv_folds}-fold CV "
                    f"(have {len(X_train_scaled)}, need {min_samples_required}). "
                    f"Skipping cross-validation."
                )
            else:
                logger.info("Performing cross-validation")
                cv_scores = cross_val_score(
                    self.model, X_train_scaled, y_train,
                    cv=cv_folds, scoring='accuracy'
                )
                logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = (train_pred == y_train).mean()

        metrics = {
            "train_accuracy": train_accuracy,
            "cv_mean": cv_scores.mean() if cv_scores is not None else None,
            "cv_std": cv_scores.std() if cv_scores is not None else None
        }

        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        return metrics

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, any]:
        """
        Evaluate classifier on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        logger.info("Evaluating on test set")

        # Scale features if needed
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        # Apply feature selection if needed
        if self.feature_selector is not None:
            X_test_scaled = self.apply_feature_selection(X_test_scaled)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)

        # Metrics
        test_accuracy = (y_pred == y_test).mean()
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(
            y_test, y_pred,
            target_names=["Legitimate", "Tampered"],
            output_dict=True
        )

        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Tampered'])}")

        return {
            "test_accuracy": test_accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "predictions": y_pred,
            "probabilities": y_proba
        }

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            feature_names: Optional list of feature names

        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Logistic Regression: use absolute coefficient magnitudes
            importances = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return None

        if feature_names is not None and len(feature_names) != len(importances):
            # Feature selection reduced dimensionality.
            # If the selector supports get_support() (SelectKBest, RFE) we can
            # recover the original names for the selected features.
            if self.feature_selector is not None and hasattr(self.feature_selector, 'get_support'):
                selected_idx = self.feature_selector.get_support(indices=True)
                if len(selected_idx) == len(importances):
                    feature_names = [feature_names[i] for i in selected_idx]
                else:
                    logger.warning(
                        f"feature_names length ({len(feature_names)}) != importances length "
                        f"({len(importances)}); get_support returned {len(selected_idx)} indices — "
                        f"falling back to generic names"
                    )
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
            else:
                logger.warning(
                    f"feature_names length ({len(feature_names)}) != importances length "
                    f"({len(importances)}) and selector has no get_support — "
                    f"falling back to generic names"
                )
                feature_names = [f"feature_{i}" for i in range(len(importances))]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        return df

    def save_model(self, save_path: Path):
        """
        Save trained model and scaler to disk.

        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model, scaler, and feature selector
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "config": self.config
        }

        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: Path):
        """
        Load trained model from disk.

        Args:
            load_path: Path to load the model from
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        model_data = joblib.load(load_path)

        self.model = model_data["model"]
        self.scaler = model_data.get("scaler")
        self.feature_selector = model_data.get("feature_selector")
        self.config = model_data.get("config", self.config)

        logger.info(f"Model loaded from {load_path}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        # Scale if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Apply feature selection if needed
        if self.feature_selector is not None:
            X_scaled = self.apply_feature_selection(X_scaled)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities
