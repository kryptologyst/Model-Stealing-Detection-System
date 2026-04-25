"""Explainability and visualization for model stealing detection."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class ModelStealingExplainer:
    """
    Explainability module for model stealing detection.
    
    This class provides various methods to explain model predictions
    and understand the factors that contribute to stealing detection.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(10)]
        self.explainer = None
        self.shap_values = None
        
    def setup_shap_explainer(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None) -> None:
        """
        Setup SHAP explainer for the model.
        
        Args:
            X_train: Training data for background
            X_test: Test data for explanation (optional)
        """
        try:
            if isinstance(self.model, RandomForestClassifier):
                self.explainer = shap.TreeExplainer(self.model)
            elif isinstance(self.model, LogisticRegression):
                self.explainer = shap.LinearExplainer(self.model, X_train)
            else:
                # Use KernelExplainer as fallback
                self.explainer = shap.KernelExplainer(self.model.predict_proba, X_train[:100])
            
            if X_test is not None:
                self.shap_values = self.explainer.shap_values(X_test)
        except Exception as e:
            print(f"Warning: Could not setup SHAP explainer: {e}")
            self.explainer = None
    
    def get_feature_importance(self, X: np.ndarray, method: str = "permutation") -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            X: Input data
            method: Method for calculating importance ('permutation', 'shap', 'model')
            
        Returns:
            Dictionary of feature importance scores
        """
        if method == "permutation":
            return self._get_permutation_importance(X)
        elif method == "shap" and self.explainer is not None:
            return self._get_shap_importance(X)
        elif method == "model" and hasattr(self.model, 'feature_importances_'):
            return self._get_model_importance()
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _get_permutation_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate permutation importance."""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model must have predict_proba method")
        
        # Get baseline score
        baseline_score = self.model.predict_proba(X)[:, 1].mean()
        
        importance_scores = {}
        for i, feature_name in enumerate(self.feature_names):
            # Permute feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate new score
            permuted_score = self.model.predict_proba(X_permuted)[:, 1].mean()
            
            # Importance is the difference
            importance_scores[feature_name] = baseline_score - permuted_score
        
        return importance_scores
    
    def _get_shap_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate SHAP importance."""
        if self.explainer is None:
            raise ValueError("SHAP explainer not setup")
        
        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Calculate mean absolute SHAP values
            importance_scores = {}
            for i, feature_name in enumerate(self.feature_names):
                importance_scores[feature_name] = np.mean(np.abs(shap_values[:, i]))
            
            return importance_scores
        except Exception as e:
            print(f"Warning: Could not calculate SHAP importance: {e}")
            return {}
    
    def _get_model_importance(self) -> Dict[str, float]:
        """Get model's built-in feature importance."""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        importance_scores = {}
        for i, feature_name in enumerate(self.feature_names):
            importance_scores[feature_name] = self.model.feature_importances_[i]
        
        return importance_scores
    
    def explain_prediction(self, query: np.ndarray, method: str = "shap") -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            query: Single query to explain
            method: Explanation method ('shap', 'permutation')
            
        Returns:
            Dictionary with explanation details
        """
        if method == "shap" and self.explainer is not None:
            return self._explain_with_shap(query)
        elif method == "permutation":
            return self._explain_with_permutation(query)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _explain_with_shap(self, query: np.ndarray) -> Dict[str, Any]:
        """Explain prediction using SHAP."""
        try:
            shap_values = self.explainer.shap_values(query.reshape(1, -1))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Get feature contributions
            contributions = {}
            for i, feature_name in enumerate(self.feature_names):
                contributions[feature_name] = float(shap_values[0, i])
            
            # Get prediction
            prediction = self.model.predict_proba(query.reshape(1, -1))[0, 1]
            
            return {
                "prediction": float(prediction),
                "contributions": contributions,
                "method": "shap"
            }
        except Exception as e:
            print(f"Warning: Could not explain with SHAP: {e}")
            return {}
    
    def _explain_with_permutation(self, query: np.ndarray) -> Dict[str, Any]:
        """Explain prediction using permutation."""
        # Get baseline prediction
        baseline_pred = self.model.predict_proba(query.reshape(1, -1))[0, 1]
        
        # Calculate feature contributions
        contributions = {}
        for i, feature_name in enumerate(self.feature_names):
            # Permute feature
            query_permuted = query.copy()
            query_permuted[i] = 0  # Set to zero as perturbation
            
            # Get new prediction
            permuted_pred = self.model.predict_proba(query_permuted.reshape(1, -1))[0, 1]
            
            # Contribution is the difference
            contributions[feature_name] = float(baseline_pred - permuted_pred)
        
        return {
            "prediction": float(baseline_pred),
            "contributions": contributions,
            "method": "permutation"
        }
    
    def get_global_explanation(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get global explanation for the model.
        
        Args:
            X: Input data for global explanation
            
        Returns:
            Dictionary with global explanation
        """
        # Get feature importance
        importance = self.get_feature_importance(X, method="permutation")
        
        # Get prediction distribution
        predictions = self.model.predict_proba(X)[:, 1]
        
        return {
            "feature_importance": importance,
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
            },
            "n_samples": len(X)
        }


class ModelStealingVisualizer:
    """
    Visualization module for model stealing detection.
    
    This class provides various visualization methods to understand
    model performance and behavior.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, title: str = "ROC Curve") -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = np.trapz(tpr, fpr)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, title: str = "Precision-Recall Curve") -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = np.trapz(precision, recall)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_feature_importance(self, importance_scores: Dict[str, float], title: str = "Feature Importance") -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_scores: Dictionary of feature importance scores
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        # Sort by importance
        sorted_indices = np.argsort(scores)[::-1]
        features = [features[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.barh(features, scores)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        return fig
    
    def plot_prediction_distribution(self, predictions: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "Prediction Distribution") -> plt.Figure:
        """
        Plot prediction distribution.
        
        Args:
            predictions: Model predictions
            labels: True labels (optional)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if labels is not None:
            # Plot distributions for each class
            normal_preds = predictions[labels == 0]
            stealing_preds = predictions[labels == 1]
            
            ax.hist(normal_preds, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
            ax.hist(stealing_preds, bins=30, alpha=0.7, label='Stealing', color='red', density=True)
            ax.legend()
        else:
            ax.hist(predictions, bins=30, alpha=0.7, color='blue', density=True)
        
        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Normal', 'Stealing'],
               yticklabels=['Normal', 'Stealing'],
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        return fig
    
    def plot_shap_summary(self, shap_values: np.ndarray, X: np.ndarray, feature_names: List[str], title: str = "SHAP Summary") -> plt.Figure:
        """
        Plot SHAP summary.
        
        Args:
            shap_values: SHAP values
            X: Input data
            feature_names: Feature names
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            fig = plt.figure(figsize=self.figsize)
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.title(title)
            return fig
        except Exception as e:
            print(f"Warning: Could not create SHAP summary plot: {e}")
            return plt.figure(figsize=self.figsize)
    
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float], title: str = "Learning Curves") -> plt.Figure:
        """
        Plot learning curves.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        epochs = range(1, len(train_scores) + 1)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(epochs, train_scores, 'b-', label='Training')
        ax.plot(epochs, val_scores, 'r-', label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def save_plots(self, plots: Dict[str, plt.Figure], output_dir: str = "assets/plots") -> None:
        """
        Save plots to files.
        
        Args:
            plots: Dictionary of plot names and figures
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in plots.items():
            fig.savefig(f"{output_dir}/{name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
