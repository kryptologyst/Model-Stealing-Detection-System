"""Evaluation metrics for model stealing detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelStealingMetrics:
    """
    Comprehensive metrics for model stealing detection evaluation.
    
    This class provides various metrics specifically designed for evaluating
    model stealing detection systems, including both traditional ML metrics
    and security-specific metrics.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of basic metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_prob is not None:
            # Use probability of positive class
            if y_prob.ndim > 1:
                y_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
            else:
                y_prob_pos = y_prob
            
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob_pos)
            metrics["pr_auc"] = average_precision_score(y_true, y_prob_pos)
        
        return metrics
    
    def calculate_security_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate security-specific metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of security metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Security-specific metrics
        metrics = {
            "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "true_negative_rate": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "positive_predictive_value": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0,
        }
        
        # Alert volume metrics
        total_queries = len(y_true)
        total_alerts = np.sum(y_pred)
        metrics["alert_rate"] = total_alerts / total_queries if total_queries > 0 else 0
        
        # Workload metrics
        metrics["alerts_per_1000_queries"] = (total_alerts / total_queries) * 1000 if total_queries > 0 else 0
        
        return metrics
    
    def calculate_precision_at_k(
        self, y_true: np.ndarray, y_prob: np.ndarray, k_values: List[int] = [10, 50, 100]
    ) -> Dict[str, float]:
        """
        Calculate precision at different k values.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary of precision@k metrics
        """
        if y_prob.ndim > 1:
            y_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
        else:
            y_prob_pos = y_prob
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(y_prob_pos)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        metrics = {}
        for k in k_values:
            if k > len(sorted_labels):
                k = len(sorted_labels)
            
            top_k_labels = sorted_labels[:k]
            precision_k = np.sum(top_k_labels) / k if k > 0 else 0
            metrics[f"precision_at_{k}"] = precision_k
        
        return metrics
    
    def calculate_recall_at_fixed_precision(
        self, y_true: np.ndarray, y_prob: np.ndarray, precision_thresholds: List[float] = [0.9, 0.95, 0.99]
    ) -> Dict[str, float]:
        """
        Calculate recall at fixed precision thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            precision_thresholds: List of precision thresholds
            
        Returns:
            Dictionary of recall@precision metrics
        """
        if y_prob.ndim > 1:
            y_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
        else:
            y_prob_pos = y_prob
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob_pos)
        
        metrics = {}
        for target_precision in precision_thresholds:
            # Find the highest recall where precision >= target_precision
            valid_indices = precision >= target_precision
            if np.any(valid_indices):
                max_recall = np.max(recall[valid_indices])
                metrics[f"recall_at_precision_{target_precision}"] = max_recall
            else:
                metrics[f"recall_at_precision_{target_precision}"] = 0.0
        
        return metrics
    
    def calculate_fpr_at_tpr(
        self, y_true: np.ndarray, y_prob: np.ndarray, tpr_thresholds: List[float] = [0.9, 0.95, 0.99]
    ) -> Dict[str, float]:
        """
        Calculate false positive rate at target true positive rates.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            tpr_thresholds: List of TPR thresholds
            
        Returns:
            Dictionary of FPR@TPR metrics
        """
        if y_prob.ndim > 1:
            y_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
        else:
            y_prob_pos = y_prob
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
        
        metrics = {}
        for target_tpr in tpr_thresholds:
            # Find the lowest FPR where TPR >= target_tpr
            valid_indices = tpr >= target_tpr
            if np.any(valid_indices):
                min_fpr = np.min(fpr[valid_indices])
                metrics[f"fpr_at_tpr_{target_tpr}"] = min_fpr
            else:
                metrics[f"fpr_at_tpr_{target_tpr}"] = 1.0
        
        return metrics
    
    def calculate_detection_delay(
        self, y_true: np.ndarray, y_pred: np.ndarray, timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate detection delay metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamps: Optional timestamps for queries
            
        Returns:
            Dictionary of detection delay metrics
        """
        if timestamps is None:
            # Use sequential indices as timestamps
            timestamps = np.arange(len(y_true))
        
        # Find first detection of each attack
        attack_indices = np.where(y_true == 1)[0]
        detection_indices = np.where(y_pred == 1)[0]
        
        if len(attack_indices) == 0:
            return {"avg_detection_delay": 0.0, "max_detection_delay": 0.0}
        
        delays = []
        for attack_idx in attack_indices:
            # Find first detection after this attack
            detections_after = detection_indices[detection_indices >= attack_idx]
            if len(detections_after) > 0:
                first_detection = detections_after[0]
                delay = timestamps[first_detection] - timestamps[attack_idx]
                delays.append(delay)
        
        if len(delays) == 0:
            return {"avg_detection_delay": float('inf'), "max_detection_delay": float('inf')}
        
        metrics = {
            "avg_detection_delay": np.mean(delays),
            "max_detection_delay": np.max(delays),
            "min_detection_delay": np.min(delays),
            "std_detection_delay": np.std(delays),
        }
        
        return metrics
    
    def calculate_all_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}
        
        # Basic metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob)
        all_metrics.update(basic_metrics)
        
        # Security metrics
        security_metrics = self.calculate_security_metrics(y_true, y_pred, y_prob)
        all_metrics.update(security_metrics)
        
        # Precision@K metrics
        if y_prob is not None:
            precision_k_metrics = self.calculate_precision_at_k(y_true, y_prob)
            all_metrics.update(precision_k_metrics)
            
            # Recall@Precision metrics
            recall_precision_metrics = self.calculate_recall_at_fixed_precision(y_true, y_prob)
            all_metrics.update(recall_precision_metrics)
            
            # FPR@TPR metrics
            fpr_tpr_metrics = self.calculate_fpr_at_tpr(y_true, y_prob)
            all_metrics.update(fpr_tpr_metrics)
        
        return all_metrics
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Create a confusion matrix DataFrame.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as DataFrame
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Stealing"],
            columns=["Predicted Normal", "Predicted Stealing"]
        )
        return cm_df
    
    def create_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Create a classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        return classification_report(y_true, y_pred, target_names=["Normal", "Stealing"])


class ModelStealingLeaderboard:
    """
    Leaderboard for model stealing detection models.
    
    This class manages a leaderboard of different detection models
    and their performance metrics.
    """
    
    def __init__(self):
        """Initialize the leaderboard."""
        self.results = []
        self.metrics_calculator = ModelStealingMetrics()
    
    def add_model_result(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a model result to the leaderboard.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            metadata: Additional metadata (optional)
        """
        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred, y_prob)
        
        # Create result entry
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "metadata": metadata or {},
            "n_samples": len(y_true),
            "n_positive": np.sum(y_true),
            "n_negative": len(y_true) - np.sum(y_true),
        }
        
        self.results.append(result)
    
    def get_leaderboard(self, sort_by: str = "f1_score") -> pd.DataFrame:
        """
        Get the leaderboard as a DataFrame.
        
        Args:
            sort_by: Metric to sort by
            
        Returns:
            Leaderboard DataFrame
        """
        if not self.results:
            return pd.DataFrame()
        
        # Extract metrics for each model
        leaderboard_data = []
        for result in self.results:
            row = {
                "model_name": result["model_name"],
                "n_samples": result["n_samples"],
                "n_positive": result["n_positive"],
                "n_negative": result["n_negative"],
            }
            row.update(result["metrics"])
            leaderboard_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by specified metric
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        
        return df
    
    def get_best_model(self, metric: str = "f1_score") -> Optional[Dict]:
        """
        Get the best model for a specific metric.
        
        Args:
            metric: Metric to optimize
            
        Returns:
            Best model result or None
        """
        if not self.results:
            return None
        
        best_result = None
        best_score = -float('inf')
        
        for result in self.results:
            if metric in result["metrics"]:
                score = result["metrics"][metric]
                if score > best_score:
                    best_score = score
                    best_result = result
        
        return best_result
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare specific models.
        
        Args:
            model_names: List of model names to compare (None for all)
            
        Returns:
            Comparison DataFrame
        """
        if model_names is None:
            model_names = [result["model_name"] for result in self.results]
        
        # Filter results
        filtered_results = [
            result for result in self.results
            if result["model_name"] in model_names
        ]
        
        if not filtered_results:
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = []
        for result in filtered_results:
            row = {"model_name": result["model_name"]}
            row.update(result["metrics"])
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_metric_summary(self, metric: str) -> Dict[str, float]:
        """
        Get summary statistics for a specific metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Summary statistics
        """
        if not self.results:
            return {}
        
        values = []
        for result in self.results:
            if metric in result["metrics"]:
                values.append(result["metrics"][metric])
        
        if not values:
            return {}
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
        }
