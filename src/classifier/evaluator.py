"""
Evaluation and visualization module for malicious LLM detector
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ClassifierEvaluator:
    """
    Evaluates and visualizes binary classifier performance.
    """

    def __init__(self, results_dir: Path):
        """
        Initialize evaluator.

        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        save_name: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix.

        Args:
            conf_matrix: Confusion matrix array
            save_name: Filename to save plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"]
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

        save_path = self.results_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"Confusion matrix saved to {save_path}")

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_name: str = "feature_importance.png"
    ):
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to plot
            save_name: Filename to save plot
        """
        if importance_df is None or len(importance_df) == 0:
            logger.warning("No feature importance data to plot")
            return

        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)

        sns.barplot(
            data=top_features,
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')

        save_path = self.results_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"Feature importance plot saved to {save_path}")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_name: str = "roc_curve.png"
    ):
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_name: Filename to save plot
        """
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        save_path = self.results_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"ROC curve saved to {save_path}")

    def save_metrics_report(
        self,
        metrics: Dict,
        save_name: str = "metrics_report.txt"
    ):
        """
        Save metrics to a text report.

        Args:
            metrics: Dictionary of metrics
            save_name: Filename to save report
        """
        save_path = self.results_dir / save_name

        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MALICIOUS LLM DETECTOR - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Training metrics
            if "train_accuracy" in metrics:
                f.write("Training Metrics:\n")
                f.write(f"  Accuracy: {metrics['train_accuracy']:.4f}\n")
                if metrics.get("cv_mean") is not None:
                    f.write(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})\n")
                f.write("\n")

            # Test metrics
            if "test_accuracy" in metrics:
                f.write("Test Metrics:\n")
                f.write(f"  Accuracy: {metrics['test_accuracy']:.4f}\n\n")

            # Confusion matrix
            if "confusion_matrix" in metrics:
                f.write("Confusion Matrix:\n")
                f.write(f"{metrics['confusion_matrix']}\n\n")

            # Classification report
            if "classification_report" in metrics:
                f.write("Classification Report:\n")
                report = metrics['classification_report']
                for label, values in report.items():
                    if isinstance(values, dict):
                        f.write(f"\n{label}:\n")
                        for metric, value in values.items():
                            f.write(f"  {metric}: {value:.4f}\n")

        logger.info(f"Metrics report saved to {save_path}")

    def create_full_report(
        self,
        train_metrics: Dict,
        test_metrics: Dict,
        importance_df: pd.DataFrame = None,
        classifier_name: str = "classifier"
    ):
        """
        Create full evaluation report with all plots and metrics.

        Args:
            train_metrics: Training metrics
            test_metrics: Test metrics
            importance_df: Feature importance dataframe
            classifier_name: Used as filename prefix to namespace outputs per classifier
        """
        logger.info(f"Creating report for: {classifier_name}")

        all_metrics = {**train_metrics, **test_metrics}

        self.save_metrics_report(all_metrics, save_name=f"{classifier_name}_metrics_report.txt")

        if "confusion_matrix" in test_metrics:
            self.plot_confusion_matrix(
                test_metrics["confusion_matrix"],
                save_name=f"{classifier_name}_confusion_matrix.png"
            )

        if importance_df is not None:
            self.plot_feature_importance(
                importance_df,
                save_name=f"{classifier_name}_feature_importance.png"
            )

        logger.info(f"Report for {classifier_name} saved to {self.results_dir}")
